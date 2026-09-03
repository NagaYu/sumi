"""Sumi 土台テキスト取得 + ライセンス台帳。

合成 PII を差し込むための「本物らしい日本語の地の文」を、**再配布可能な
ライセンスの出典だけ**から集めてキャッシュする層。

設計方針:

* 取得は 3 系統 (ウィキペディア日本語版 / e-Gov 法令 API / 青空文庫) のみ。
  いずれも CC BY-SA もしくはパブリックドメインで、出典表記さえ守れば
  データセットとして再配布できる。
* 取得物は必ず ``data/raw/<source>.jsonl`` に追記し、**2 回目以降は完全に
  オフラインで同じコーパスが再現できる**。ネットワーク断でも例外は投げず、
  キャッシュ (無ければ空リスト) に縮退する。
* 本文は段落単位で 120〜600 字程度のチャンクに割り、markup / 表組み /
  日本語が 40 字未満の断片は捨てる。
* すべてのチャンクに :func:`sumi.types.normalize` を **ちょうど 1 回** だけ
  適用してからキャッシュへ書く。以後 (合成 PII の挿入時) は再正規化しない。
  再正規化するとスパン座標が壊れるため。

Claim: 検出率 / 低誤検出 — 評価用の地の文が実文書の分布 (百科事典・法令・
文芸) に近いほど、報告する検出率と誤検出率が実運用の値に近づく。
出典とライセンスを台帳として持つことで、データセットカードに貼れる形で
再現性と再配布可能性を担保する。
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, Sequence

from .types import normalize

__all__ = [
    "CorpusItem",
    "LICENSES",
    "EGOV_LAW_IDS",
    "USER_AGENT",
    "fetch_wikipedia",
    "fetch_egov_laws",
    "fetch_aozora",
    "load_base_corpus",
    "license_table",
    "chunk_text",
    "cache_path",
    "read_cache",
    "append_cache",
    "net_stats",
    "is_offline",
]

# --------------------------------------------------------------------------
# 定数
# --------------------------------------------------------------------------

#: ウィキペディア API は「連絡先の分かる説明的な User-Agent」を要求する。
USER_AGENT = (
    "SumiPIICorpus/0.1 (Japanese PII-detection research corpus builder; "
    "non-commercial, low-volume, cached) python-urllib/3"
)

#: 環境変数でネットワークを完全に切る (キャッシュだけで動かす) ためのフラグ。
OFFLINE_ENV = "SUMI_OFFLINE"

MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = 600
MIN_JA_CHARS = 40           # 日本語文字がこれ未満のチャンクは捨てる
MIN_JA_RATIO = 0.30         # 日本語文字比率の下限 (数表・英字羅列を排除)
MAX_MARKUP_RATIO = 0.04     # 記号 (=|{}[]<>#*) 比率の上限
MAX_CHUNKS_PER_DOC = 4      # 同一ページ/作品からの偏りを避ける
_POLITE_DELAY = 0.34        # 連続リクエスト間隔 (秒)
_HTTP_TIMEOUT = 45.0

#: ライセンス台帳。データセットカードにそのまま貼れる粒度で持つ。
LICENSES: dict[str, dict] = {
    "wikipedia_ja": {
        "source_ja": "ウィキペディア日本語版",
        "source_en": "Japanese Wikipedia",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "url": "https://ja.wikipedia.org/",
        "api": "https://ja.wikipedia.org/w/api.php",
        "genre": "wiki",
        "share_alike": True,
        "attribution_template": "ウィキペディア日本語版「{title}」 (CC BY-SA 4.0)",
        "note": (
            "Article text is CC BY-SA 4.0. Redistribution inherits the same licence; "
            "the article title and the modifications made (paragraph splitting, NFKC "
            "normalisation, insertion of synthetic PII) are disclosed."
        ),
    },
    "egov_law": {
        "source_ja": "e-Gov 法令検索 (法令 API v2)",
        "source_en": "e-Gov Japanese statute search (Law API v2)",
        "license": "Public Domain (Japanese Copyright Act, Article 13)",
        "license_url": "https://elaws.e-gov.go.jp/",
        "url": "https://laws.e-gov.go.jp/",
        "api": "https://laws.e-gov.go.jp/api/2/law_data/{law_id}?response_format=json",
        "genre": "law",
        "share_alike": False,
        "attribution_template": "e-Gov法令検索『{title}』{article} (著作権法第13条により著作権の目的とならない)",
        "note": (
            "Statutes and official notices are not subject to copyright under Article 13 "
            "of the Japanese Copyright Act. Only article body text is extracted; tables of "
            "contents, supplementary provisions and appended tables are excluded."
        ),
    },
    "aozora": {
        "source_ja": "青空文庫 (globis-university/aozorabunko-clean)",
        "source_en": "Aozora Bunko (via globis-university/aozorabunko-clean)",
        "license": "Public Domain (Aozora Bunko, copyright expired)",
        "license_url": "https://www.aozora.gr.jp/guide/kijyunn.html",
        "url": "https://www.aozora.gr.jp/",
        "api": "hf://datasets/globis-university/aozorabunko-clean",
        "genre": "novel",
        "share_alike": False,
        "attribution_template": "青空文庫『{title}』{author} (著作権保護期間満了)",
        "note": (
            "著作権保護期間が満了した作品のみを収録した clean 版を streaming 取得。"
            "作品著作権フラグ・人物著作権フラグが「あり」の記録は使用しない。"
        ),
    },
}

#: ``sources=`` に渡せる短縮名 -> ライセンス台帳キー。
SOURCE_ALIASES: dict[str, str] = {
    "wikipedia": "wikipedia_ja",
    "wikipedia_ja": "wikipedia_ja",
    "wiki": "wikipedia_ja",
    "egov": "egov_law",
    "egov_law": "egov_law",
    "law": "egov_law",
    "aozora": "aozora",
    "novel": "aozora",
}

#: e-Gov 法令 API v2 で解決することを実際に確認済みの法令 ID (33件)。
#: (law_id, 通称) — 通称は attribution の可読性のためだけに持つ。
EGOV_LAW_IDS: tuple[tuple[str, str], ...] = (
    ("321CONSTITUTION", "日本国憲法"),
    ("129AC0000000089", "民法"),
    ("132AC0000000048", "商法"),
    ("140AC0000000045", "刑法"),
    ("322AC0000000026", "学校教育法"),
    ("322AC0000000049", "労働基準法"),
    ("322AC0000000054", "独占禁止法"),
    ("322AC0000000067", "地方自治法"),
    ("322AC0000000120", "国家公務員法"),
    ("323AC0000000025", "金融商品取引法"),
    ("323AC0000000131", "刑事訴訟法"),
    ("323AC0000000205", "医療法"),
    ("334AC0000000121", "特許法"),
    ("335AC0000000105", "道路交通法"),
    ("345AC0000000048", "著作権法"),
    ("347AC0000000057", "労働安全衛生法"),
    ("354AC0000000004", "民事執行法"),
    ("356AC0000000059", "銀行法"),
    ("359AC0000000086", "電気通信事業法"),
    ("360AC0000000088", "労働者派遣法"),
    ("405AC0000000047", "不正競争防止法"),
    ("405AC0000000088", "行政手続法"),
    ("408AC0000000109", "民事訴訟法"),
    ("411AC0000000042", "行政機関情報公開法"),
    ("412AC0000000061", "消費者契約法"),
    ("415AC0000000057", "個人情報保護法"),
    ("416AC0000000075", "破産法"),
    ("417AC0000000086", "会社法"),
    ("418AC0000000120", "教育基本法"),
    ("419AC0000000128", "労働契約法"),
    ("425AC0000000027", "マイナンバー法"),
    ("426AC0000000068", "行政不服審査法"),
    ("503AC0000000035", "デジタル社会形成基本法"),
)

_NET_STATS: dict[str, int] = {"requests": 0, "failures": 0, "bytes": 0}


# --------------------------------------------------------------------------
# データ構造
# --------------------------------------------------------------------------


@dataclass
class CorpusItem:
    """PII を差し込む前の、ライセンスの分かっている地の文 1 チャンク。

    Claim: 検出率 — 合成文書の「土台」を出典つきで保持することで、
    検出率をジャンル (百科事典 / 法令 / 文芸) 別に層別集計できる。

    Attributes:
        text: NFKC 正規化済みの本文 (:func:`sumi.types.normalize` を 1 回適用済み)。
        license: ライセンス表記 (例 ``"CC BY-SA 4.0"``)。
        source: 台帳キー (``wikipedia_ja`` / ``egov_law`` / ``aozora``)。
        genre: ``wiki`` / ``law`` / ``novel``。
        attribution: 出典表示に使う 1 行文字列。
        item_id: ``<source>:<doc>:<chunk>`` 形式の一意 ID (キャッシュの重複排除キー)。
    """

    text: str
    license: str
    source: str
    genre: str
    attribution: str
    item_id: str

    def to_dict(self) -> dict[str, Any]:
        """JSONL キャッシュ 1 行分の dict を返す。

        Claim: 検出率 — キャッシュ往復で本文が変化しないこと (再正規化しないこと)
        が、オフライン再実行時に同一コーパス・同一スパン座標を再現する条件。
        """
        d = asdict(self)
        d["normalized"] = True  # 読み戻し時に再正規化してはならない印
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CorpusItem":
        """:meth:`to_dict` の逆変換。**text は再正規化しない**。

        Claim: 可逆性 — 文字オフセットは正規化済み本文に対して定義されるため、
        読み戻しで正規化を重ねないことがスパン座標の不変条件になる。
        """
        return CorpusItem(
            text=d["text"],
            license=d.get("license", ""),
            source=d.get("source", ""),
            genre=d.get("genre", ""),
            attribution=d.get("attribution", ""),
            item_id=d.get("item_id", ""),
        )


# --------------------------------------------------------------------------
# 汎用ユーティリティ
# --------------------------------------------------------------------------

_JA_RE = re.compile(r"[々ぁ-ゟ゠-ヿ㐀-䶿一-鿿]")
_MARKUP_RE = re.compile(r"[=|{}\[\]<>#*_~^\\]")
_HEADING_RE = re.compile(r"^\s*=+[^=]*=+\s*$")
_SENT_SPLIT_RE = re.compile(r"(?<=[。．！？!?])")
_WS_RE = re.compile(r"[ \t　]+")

#: ウィキペディア本文でここ以降は本文とみなさない節見出し。
_DROP_SECTIONS = (
    "脚注", "注釈", "出典", "参考文献", "関連項目", "外部リンク", "参照",
    "ギャラリー", "参考", "文献", "註", "補注", "リンク", "作品一覧",
    "ディスコグラフィ", "受賞", "参考資料",
)


def is_offline() -> bool:
    """環境変数 ``SUMI_OFFLINE`` によりネットワーク取得を止めるか。

    Claim: CPU速度 — ベンチマーク実行中に予期しないネットワーク待ちが混ざると
    計測値が汚れるため、明示的にオフラインへ固定できるようにする。
    """
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def net_stats() -> dict[str, int]:
    """このプロセスで発生した HTTP リクエスト数・失敗数・受信バイト数。

    Claim: CPU速度 — 「2 回目の実行はキャッシュだけで動く」ことを、
    リクエスト数 0 という観測可能な事実として自己テストで示すため。
    """
    return dict(_NET_STATS)


def _ja_chars(s: str) -> int:
    return len(_JA_RE.findall(s))


def _http_get_json(url: str, *, timeout: float = _HTTP_TIMEOUT, retries: int = 2) -> dict | None:
    """JSON を GET する。**失敗しても例外を投げず None を返す**。

    Claim: 低誤検出 — 取得失敗時に静かにキャッシュへ縮退することで、
    ネットワーク事情によって評価データが黙って変質する事故を避ける。
    """
    if is_offline():
        return None
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "application/json"})
            _NET_STATS["requests"] += 1
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            _NET_STATS["bytes"] += len(raw)
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as exc:          # noqa: BLE001 — ネットワークは何でも起きる
            last_err = exc
            _NET_STATS["failures"] += 1
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
    if last_err is not None:
        _dbg(f"HTTP failed: {url[:90]} -> {type(last_err).__name__}")
    return None


def _dbg(msg: str) -> None:
    if os.environ.get("SUMI_VERBOSE"):
        print(f"[corpus] {msg}")


# --------------------------------------------------------------------------
# チャンク分割
# --------------------------------------------------------------------------


def _looks_like_markup_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith(("|", "!", "*", "#", ":", ";", "{", "}")):
        return True
    if s.count("|") >= 2 or "{{" in s or "}}" in s or "[[" in s:
        return True
    return False


def _content_paragraphs(text: str) -> list[str]:
    """生テキストを「本文っぽい段落」の列に整形する (見出し・表組みは捨てる)。"""
    paras: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            p = "\n".join(buf).strip()
            if p:
                paras.append(p)
            buf.clear()

    for raw_line in text.split("\n"):
        line = _WS_RE.sub(" ", raw_line).strip()
        if not line:
            flush()
            continue
        # 判定だけ NFKC 済みの写しで行う (全角 ＝ ｜ の markup を取りこぼさないため)。
        # 出力するのは raw のままの line で、normalize() は chunk_text の最後に 1 回だけ。
        probe = unicodedata.normalize("NFKC", line)
        if _HEADING_RE.match(probe):
            title = probe.strip("= ").strip()
            flush()
            if any(title.startswith(d) or title == d for d in _DROP_SECTIONS):
                break  # 脚注以降は本文ではない
            continue
        if _looks_like_markup_line(probe):
            flush()
            continue
        buf.append(line)
    flush()
    return paras


def _split_long(paragraph: str, min_chars: int, max_chars: int) -> list[str]:
    """長すぎる段落を文境界で max_chars 以下に割る。"""
    pieces = [p for p in _SENT_SPLIT_RE.split(paragraph) if p]
    out: list[str] = []
    buf = ""
    for piece in pieces:
        while len(piece) > max_chars:          # 句点の無い長大文への保険
            if buf:
                out.append(buf)
                buf = ""
            out.append(piece[:max_chars])
            piece = piece[max_chars:]
        if buf and len(buf) + len(piece) > max_chars:
            out.append(buf)
            buf = piece
        else:
            buf += piece
        if len(buf) >= min_chars and len(buf) >= (min_chars + max_chars) // 2:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def _is_good_chunk(chunk: str) -> bool:
    n = len(chunk)
    if n < MIN_CHUNK_CHARS // 2:
        return False
    ja = _ja_chars(chunk)
    if ja < MIN_JA_CHARS:
        return False
    if ja / n < MIN_JA_RATIO:
        return False
    if len(_MARKUP_RE.findall(chunk)) / n > MAX_MARKUP_RATIO:
        return False
    lines = [l for l in chunk.split("\n") if l.strip()]
    if len(lines) >= 4 and sum(1 for l in lines if len(l) < 14) / len(lines) > 0.6:
        return False  # 箇条書き・年表のような列挙
    return True


def chunk_text(
    text: str,
    *,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    max_chunks: int | None = None,
) -> list[str]:
    """生テキストを 120〜600 字程度の本文チャンク列にし、正規化して返す。

    見出し・表組み・箇条書き・markup 過多・日本語 40 字未満の断片は捨てる。
    :func:`sumi.types.normalize` は各チャンクに **ちょうど 1 回** だけ適用する。

    Claim: 検出率 / 低誤検出 — 合成 PII を差し込む土台の粒度を業務文書 1 段落
    程度に揃えることで、モデルの文脈長に収まり、かつ「文脈語が近くにある/ない」
    条件を現実的な割合に保つ。表組み等の非文を除くことで、地の文由来の
    無意味な誤検出を評価から排除する。
    """
    if not text:
        return []
    paragraphs = _content_paragraphs(text)
    packed: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(p) > max_chars:
            if buf:
                packed.append(buf)
                buf = ""
            packed.extend(_split_long(p, min_chars, max_chars))
            continue
        if buf and len(buf) + 1 + len(p) > max_chars:
            packed.append(buf)
            buf = p
        else:
            buf = f"{buf}\n{p}" if buf else p
        if len(buf) >= max_chars:
            packed.append(buf)
            buf = ""
    if buf:
        packed.append(buf)

    out: list[str] = []
    for c in packed:
        c = c.strip()
        if len(c) < min_chars:
            continue
        c = normalize(c)          # ← 正規化はここで 1 回だけ
        if not _is_good_chunk(c):
            continue
        out.append(c)
        if max_chunks is not None and len(out) >= max_chunks:
            break
    return out


# --------------------------------------------------------------------------
# キャッシュ (data/raw/<source>.jsonl)
# --------------------------------------------------------------------------


def cache_path(source: str, *, cache_dir: str = "data/raw") -> str:
    """``data/raw/<source>.jsonl`` の絶対でない実パスを返す。

    Claim: CPU速度 — 取得済みコーパスをローカル JSONL に固定することで、
    ベンチマークの再実行がネットワーク待ちなしで走る。
    """
    return os.path.join(cache_dir, f"{_canonical_source(source)}.jsonl")


def read_cache(source: str, *, cache_dir: str = "data/raw") -> list[CorpusItem]:
    """キャッシュ JSONL を読み、壊れた行は黙って飛ばして返す。

    Claim: 検出率 — 同じキャッシュから同じ土台コーパスが再現できることが、
    報告した検出率を第三者が再計算できる条件。
    """
    path = cache_path(source, cache_dir=cache_dir)
    items: list[CorpusItem] = []
    if not os.path.exists(path):
        return items
    seen: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item_id = d.get("item_id") or ""
                if not item_id or item_id in seen:
                    continue
                if not d.get("text"):
                    continue
                seen.add(item_id)
                items.append(CorpusItem.from_dict(d))
    except OSError:
        return items
    return items


def append_cache(items: Sequence[CorpusItem], source: str, *, cache_dir: str = "data/raw") -> int:
    """新規 ``item_id`` だけを JSONL へ追記し、書いた件数を返す (追記安全)。

    Claim: 検出率 — 取得を何度に分けても同一 ID が重複しないため、
    コーパス件数と検出率の分母が実行のたびにぶれない。
    """
    if not items:
        return 0
    path = cache_path(source, cache_dir=cache_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    existing = {it.item_id for it in read_cache(source, cache_dir=cache_dir)}
    written = 0
    try:
        with open(path, "a", encoding="utf-8") as fh:
            for it in items:
                if not it.item_id or it.item_id in existing:
                    continue
                existing.add(it.item_id)
                fh.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
                written += 1
            fh.flush()
    except OSError as exc:
        _dbg(f"cache write failed: {exc}")
    return written


def _canonical_source(source: str) -> str:
    key = SOURCE_ALIASES.get(source, source)
    if key not in LICENSES:
        raise ValueError(f"unknown corpus source: {source!r} (known: {sorted(LICENSES)})")
    return key


def _sample(pool: Sequence[CorpusItem], k: int, seed: int) -> list[CorpusItem]:
    """item_id 順に固定してから種つき乱数で k 件選ぶ (実行順に依存しない)。"""
    ordered = sorted(pool, key=lambda it: it.item_id)
    if k >= len(ordered):
        return ordered
    rng = random.Random(seed)
    picked = rng.sample(ordered, k)
    return sorted(picked, key=lambda it: it.item_id)


# --------------------------------------------------------------------------
# 1) ウィキペディア日本語版
# --------------------------------------------------------------------------

_WIKI_API = "https://ja.wikipedia.org/w/api.php"


def _wiki_pages(payload: dict) -> list[dict]:
    query = payload.get("query") or {}
    pages = query.get("pages")
    if isinstance(pages, dict):
        return list(pages.values())
    if isinstance(pages, list):
        return pages
    return []


def fetch_wikipedia(
    n: int, *, seed: int = 0, cache_dir: str = "data/raw", min_chars: int = MIN_CHUNK_CHARS
) -> list[CorpusItem]:
    """ウィキペディア日本語版からランダム記事本文チャンクを n 件集める。

    ``action=query&generator=random&prop=extracts&explaintext=1`` を、n 件
    そろうまで繰り返し叩く (1 回あたり最大 20 記事)。取得済みはキャッシュから
    先に埋めるので、2 回目以降はネットワークに触らない。ネットワーク失敗時は
    例外を投げずキャッシュ (無ければ空リスト) を返す。

    Claim: 検出率 / 低誤検出 — 百科事典の地の文は「人名・地名・数字が自然に
    出てくるが PII ではない」文の宝庫であり、正解 0 件の否定例土台としても、
    合成 PII を埋める土台としても、実運用に近い誤検出圧力を与える。
    """
    key = "wikipedia_ja"
    if n <= 0:
        return []
    cached = read_cache(key, cache_dir=cache_dir)
    if len(cached) >= n or is_offline():
        return _sample(cached, min(n, len(cached)), seed)

    need = n - len(cached)
    seen = {it.item_id for it in cached}
    lic = LICENSES[key]
    fresh: list[CorpusItem] = []
    fails = 0
    rounds = 0
    max_rounds = 40

    while len(fresh) < need and rounds < max_rounds and fails < 4:
        rounds += 1
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": 20,
            "prop": "extracts",
            "explaintext": 1,
            "exlimit": 20,
        }
        payload = _http_get_json(_WIKI_API + "?" + urllib.parse.urlencode(params))
        if payload is None:
            fails += 1
            continue
        got_this_round = 0
        for page in _wiki_pages(payload):
            extract = page.get("extract") or ""
            title = page.get("title") or ""
            pageid = page.get("pageid")
            if not extract or pageid is None:
                continue
            chunks = chunk_text(extract, min_chars=min_chars, max_chunks=MAX_CHUNKS_PER_DOC)
            for idx, chunk in enumerate(chunks):
                item_id = f"{key}:{pageid}:{idx}"
                if item_id in seen:
                    continue
                seen.add(item_id)
                fresh.append(
                    CorpusItem(
                        text=chunk,
                        license=lic["license"],
                        source=key,
                        genre=lic["genre"],
                        attribution=lic["attribution_template"].format(title=title),
                        item_id=item_id,
                    )
                )
                got_this_round += 1
                if len(fresh) >= need:
                    break
            if len(fresh) >= need:
                break
        _dbg(f"wikipedia round {rounds}: +{got_this_round} (total fresh {len(fresh)})")
        if len(fresh) < need:
            time.sleep(_POLITE_DELAY)

    if fresh:
        append_cache(fresh, key, cache_dir=cache_dir)
    pool = cached + fresh
    return _sample(pool, min(n, len(pool)), seed)


# --------------------------------------------------------------------------
# 2) e-Gov 法令 API v2
# --------------------------------------------------------------------------

_EGOV_URL = "https://laws.e-gov.go.jp/api/2/law_data/{law_id}?response_format=json"

#: 条文本文ではない枝 (目次・附則・別表・様式) は歩かない。
_LAW_SKIP_TAGS = frozenset(
    {
        "TOC", "SupplProvision", "AppdxTable", "AppdxNote", "AppdxStyle",
        "AppdxFig", "AppdxFormat", "Appdx", "LawNum", "LawTitle",
        "EnactStatement", "TableStruct", "Table", "FigStruct", "Fig",
        "Remarks", "NoteStruct", "Note", "StyleStruct", "Style", "FormatStruct",
        "ArithFormula", "Ruby", "Rt",
    }
)
_LAW_BREAK_AFTER = frozenset(
    {"ArticleCaption", "Paragraph", "Item", "Subitem1", "Subitem2", "ChapterTitle", "SectionTitle"}
)
_LAW_SPACE_AFTER = frozenset(
    {"ArticleTitle", "ParagraphNum", "ItemTitle", "Subitem1Title", "Subitem2Title", "ColumnNum"}
)


def _law_node_text(node: Any) -> str:
    """law_full_text ツリーを再帰的に歩いて条文文字列を組み立てる。"""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    tag = node.get("tag") or ""
    if tag in _LAW_SKIP_TAGS:
        return ""
    parts = [_law_node_text(c) for c in (node.get("children") or [])]
    text = "".join(parts)
    if not text:
        return ""
    if tag in _LAW_BREAK_AFTER:
        text += "\n"
    elif tag in _LAW_SPACE_AFTER:
        text += "　"
    return text


def _law_articles(node: Any, out: list[tuple[str, str]]) -> None:
    """``Article`` 単位で (条番号, 本文) を収集する。"""
    if not isinstance(node, dict):
        return
    tag = node.get("tag") or ""
    if tag in _LAW_SKIP_TAGS:
        return
    if tag == "Article":
        num = str((node.get("attr") or {}).get("Num", len(out) + 1))
        body = _law_node_text(node).strip()
        if body:
            out.append((num, body))
        return
    for child in node.get("children") or []:
        _law_articles(child, out)


def fetch_egov_laws(n: int, *, seed: int = 0, cache_dir: str = "data/raw") -> list[CorpusItem]:
    """e-Gov 法令 API v2 から条文チャンクを n 件集める。

    :data:`EGOV_LAW_IDS` (実際に解決することを確認済みの 33 法令) を種つき乱数で
    並べ替えて順に取得し、``law_full_text`` を再帰的に歩いて条単位の本文を取り出す。
    法令は著作権法第13条により著作権の目的とならないため、ライセンスは
    ``"Public Domain (著作権法第13条)"``。

    Claim: 低誤検出 — 法令文は「第三十二条」「五年以下」「一〇〇万円」のような
    数字表現と、「氏名」「住所」「生年月日」という語そのものが高密度で現れる。
    PII が 1 件も無いのに PII らしい形をした文の代表例であり、誤検出率の
    厳しい試験土台になる。
    """
    key = "egov_law"
    if n <= 0:
        return []
    cached = read_cache(key, cache_dir=cache_dir)
    if len(cached) >= n or is_offline():
        return _sample(cached, min(n, len(cached)), seed)

    need = n - len(cached)
    seen = {it.item_id for it in cached}
    lic = LICENSES[key]
    rng = random.Random(seed)
    law_ids = list(EGOV_LAW_IDS)
    rng.shuffle(law_ids)

    fresh: list[CorpusItem] = []
    fails = 0
    for law_id, nickname in law_ids:
        if len(fresh) >= need or fails >= 3:
            break
        payload = _http_get_json(_EGOV_URL.format(law_id=law_id))
        if payload is None:
            fails += 1
            continue
        rev = payload.get("revision_info") or {}
        title = rev.get("law_title") or nickname
        law_num = (payload.get("law_info") or {}).get("law_num") or ""
        articles: list[tuple[str, str]] = []
        _law_articles(payload.get("law_full_text"), articles)
        if not articles:
            continue
        rng.shuffle(articles)
        taken = 0
        for art_num, body in articles:
            if taken >= max(2, MAX_CHUNKS_PER_DOC) or len(fresh) >= need:
                break
            chunks = chunk_text(body, max_chunks=2)
            for idx, chunk in enumerate(chunks):
                item_id = f"{key}:{law_id}:{art_num}:{idx}"
                if item_id in seen:
                    continue
                seen.add(item_id)
                fresh.append(
                    CorpusItem(
                        text=chunk,
                        license=lic["license"],
                        source=key,
                        genre=lic["genre"],
                        attribution=lic["attribution_template"].format(
                            title=f"{title}（{law_num}）" if law_num else title,
                            article=f"第{art_num}条",
                        ),
                        item_id=item_id,
                    )
                )
                taken += 1
                if len(fresh) >= need:
                    break
        _dbg(f"egov {law_id} ({nickname}): articles={len(articles)} fresh={len(fresh)}")
        time.sleep(_POLITE_DELAY)

    if fresh:
        append_cache(fresh, key, cache_dir=cache_dir)
    pool = cached + fresh
    return _sample(pool, min(n, len(pool)), seed)


# --------------------------------------------------------------------------
# 3) 青空文庫 (HuggingFace datasets streaming)
# --------------------------------------------------------------------------

_AOZORA_DATASET = "globis-university/aozorabunko-clean"
_AOZORA_MAX_RECORDS = 60          # streaming で開く記録数の上限
_AOZORA_DEADLINE_SEC = 180.0      # これを超えたら打ち切ってキャッシュへ縮退


def fetch_aozora(n: int, *, seed: int = 0, cache_dir: str = "data/raw") -> list[CorpusItem]:
    """青空文庫 (clean 版) を streaming で開き、本文チャンクを n 件集める。

    HuggingFace ``datasets`` の streaming は初回のオープンが重いので、
    読む記録数と経過時間の両方に上限を置き、``datasets`` が無い/失敗する環境では
    例外を出さずキャッシュ (無ければ空リスト) に縮退する。

    Claim: 検出率 — 文芸作品の地の文は敬称つき人名 (「田中さん」「山田氏」) や
    旧字体・和暦・住所らしい地名が自然文の中に現れるため、
    NAME/ADDRESS/DOB の境界判定を鍛える土台になる。
    """
    key = "aozora"
    if n <= 0:
        return []
    cached = read_cache(key, cache_dir=cache_dir)
    if len(cached) >= n or is_offline():
        return _sample(cached, min(n, len(cached)), seed)

    need = n - len(cached)
    seen = {it.item_id for it in cached}
    lic = LICENSES[key]
    fresh: list[CorpusItem] = []
    started = time.time()

    try:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from datasets import load_dataset  # type: ignore

        try:
            from datasets.utils.logging import disable_progress_bar  # type: ignore

            disable_progress_bar()
        except Exception:  # noqa: BLE001
            pass

        _NET_STATS["requests"] += 1
        ds = load_dataset(_AOZORA_DATASET, split="train", streaming=True)
        rng = random.Random(seed)
        max_records = min(_AOZORA_MAX_RECORDS, max(4, need * 2))
        n_read = 0
        for rec in ds:
            if len(fresh) >= need or n_read >= max_records:
                break
            if time.time() - started > _AOZORA_DEADLINE_SEC:
                _dbg("aozora: deadline reached, stopping early")
                break
            n_read += 1
            meta = rec.get("meta") or {}
            if str(meta.get("作品著作権フラグ", "なし")) not in ("なし", "", "None"):
                continue
            if str(meta.get("人物著作権フラグ", "なし")) not in ("なし", "", "None"):
                continue
            work_id = str(meta.get("作品ID") or n_read)
            title = str(meta.get("作品名") or "")
            author = f"{meta.get('姓') or ''}{meta.get('名') or ''}".strip()
            body = rec.get("text") or ""
            chunks = chunk_text(body, max_chunks=MAX_CHUNKS_PER_DOC * 3)
            if not chunks:
                continue
            # 冒頭 (題辞・詩) を避け、作品内から散らして取る
            picks = chunks if len(chunks) <= MAX_CHUNKS_PER_DOC else rng.sample(
                chunks[1:] or chunks, MAX_CHUNKS_PER_DOC
            )
            for chunk in picks:
                idx = chunks.index(chunk)
                item_id = f"{key}:{work_id}:{idx}"
                if item_id in seen:
                    continue
                seen.add(item_id)
                fresh.append(
                    CorpusItem(
                        text=chunk,
                        license=lic["license"],
                        source=key,
                        genre=lic["genre"],
                        attribution=lic["attribution_template"].format(
                            title=title, author=author
                        ),
                        item_id=item_id,
                    )
                )
                if len(fresh) >= need:
                    break
        _dbg(f"aozora: read {n_read} records, {len(fresh)} fresh chunks "
             f"in {time.time() - started:.1f}s")
    except Exception as exc:  # noqa: BLE001 — datasets 未導入 / 通信断 / スキーマ変更
        _NET_STATS["failures"] += 1
        _dbg(f"aozora unavailable: {type(exc).__name__}: {exc}")

    if fresh:
        append_cache(fresh, key, cache_dir=cache_dir)
    pool = cached + fresh
    return _sample(pool, min(n, len(pool)), seed)


# --------------------------------------------------------------------------
# まとめ取得 + 台帳
# --------------------------------------------------------------------------

_FETCHERS: dict[str, Callable[..., list[CorpusItem]]] = {
    "wikipedia_ja": fetch_wikipedia,
    "egov_law": fetch_egov_laws,
    "aozora": fetch_aozora,
}


def load_base_corpus(
    n: int,
    *,
    seed: int = 0,
    cache_dir: str = "data/raw",
    sources: Sequence[str] = ("wikipedia", "egov", "aozora"),
) -> list[CorpusItem]:
    """3 系統から土台テキストを合計 n 件そろえる (取得できたものだけで動く)。

    各系統に均等な取り分を割り当て、足りない分は成功した系統から 1 度だけ
    追加取得して補う。並び順は ``seed`` で決まる決定論的シャッフル。

    Claim: 検出率 / 低誤検出 — ジャンルを混ぜた土台に対して同じ合成手順を
    適用することで、検出率がジャンル依存でどう動くかを一つの実験で測れる。
    どれか 1 系統が落ちても評価が止まらないことは、再現実験の実用条件。
    """
    if n <= 0:
        return []
    keys: list[str] = []
    for s in sources:
        k = _canonical_source(s)
        if k not in keys:
            keys.append(k)
    if not keys:
        return []

    per = -(-n // len(keys))  # ceil
    pool: list[CorpusItem] = []
    seen: set[str] = set()
    ok_keys: list[str] = []
    for i, k in enumerate(keys):
        got = _FETCHERS[k](per, seed=seed + 1009 * i, cache_dir=cache_dir)
        if got:
            ok_keys.append(k)
        for it in got:
            if it.item_id not in seen:
                seen.add(it.item_id)
                pool.append(it)

    if len(pool) < n and ok_keys:
        deficit = n - len(pool)
        for i, k in enumerate(ok_keys):
            if deficit <= 0:
                break
            extra = _FETCHERS[k](per + deficit, seed=seed + 7919 * (i + 1), cache_dir=cache_dir)
            for it in extra:
                if it.item_id not in seen:
                    seen.add(it.item_id)
                    pool.append(it)
                    deficit -= 1
                    if deficit <= 0:
                        break

    pool.sort(key=lambda it: it.item_id)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def license_table() -> list[dict]:
    """データセットカードに貼れるライセンス表 (1 系統 1 行) を返す。

    Claim: 可逆性 / 低誤検出 — 出典とライセンスを機械可読な形で持ち回ることで、
    生成物の再配布可否をデータセット側で判定でき、
    「配布できない素材が混ざる」事故を構造的に防ぐ。
    """
    rows: list[dict] = []
    for key, info in LICENSES.items():
        rows.append(
            {
                "source": key,
                "出典": info["source_ja"],
                "license": info["license"],
                "license_url": info["license_url"],
                "url": info["url"],
                "genre": info["genre"],
                "継承条件": "あり (SA)" if info["share_alike"] else "なし",
                "attribution": info["attribution_template"],
                "備考": info["note"],
            }
        )
    return rows


# --------------------------------------------------------------------------
# 自己テスト
# --------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    CACHE = os.environ.get("SUMI_CACHE_DIR", "data/raw")
    print("=" * 72)
    print("sumi.corpus 自己テスト  (cache_dir =", CACHE, ")")
    print("=" * 72)

    # --- 1. チャンク分割の単体確認 (ネットワーク不要) -------------------
    sample_raw = (
        "== 概要 ==\n"
        "｜表｜組み｜の｜行｜は｜落とす｜\n"
        "これはテスト用の段落です。" + "日本語の文をいくつか並べて、"
        "チャンク分割が段落単位で行われることを確認します。" * 4
        + "\n\n== 脚注 ==\nここから先は本文ではないので捨てられる。"
    )
    chunks = chunk_text(sample_raw)
    print(f"[chunk] 入力 {len(sample_raw)}字 -> {len(chunks)}チャンク "
          f"(長さ {[len(c) for c in chunks]})")
    assert chunks, "チャンクが 1 件も出ていない"
    assert all(MIN_CHUNK_CHARS <= len(c) <= MAX_CHUNK_CHARS for c in chunks), \
        [len(c) for c in chunks]
    assert "脚注" not in "".join(chunks) and "表" not in "".join(chunks)
    assert all(c == normalize(c) for c in chunks), "normalize が冪等でない"
    print(f"[chunk] 先頭: {chunks[0][:60]}…")

    # --- 2. 各ソースを少量取得 -------------------------------------------
    before = net_stats()
    per_source: dict[str, list[CorpusItem]] = {}
    t0 = time.time()
    per_source["wikipedia_ja"] = fetch_wikipedia(6, seed=7, cache_dir=CACHE)
    t_wiki = time.time() - t0
    t0 = time.time()
    per_source["egov_law"] = fetch_egov_laws(6, seed=7, cache_dir=CACHE)
    t_egov = time.time() - t0
    t0 = time.time()
    per_source["aozora"] = fetch_aozora(4, seed=7, cache_dir=CACHE)
    t_aozora = time.time() - t0

    print("-" * 72)
    print(f"{'source':<14}{'件数':>4}  {'平均字数':>8}  license")
    for key, items in per_source.items():
        avg = sum(len(i.text) for i in items) / len(items) if items else 0
        lic = LICENSES[key]["license"]
        print(f"{key:<14}{len(items):>4}  {avg:>8.0f}  {lic}")
    print(f"取得時間: wiki {t_wiki:.1f}s / egov {t_egov:.1f}s / aozora {t_aozora:.1f}s")
    after = net_stats()
    print(f"HTTP: requests {after['requests'] - before['requests']}, "
          f"failures {after['failures'] - before['failures']}, "
          f"{(after['bytes'] - before['bytes']) / 1024:.0f} KiB")

    for key, items in per_source.items():
        if items:
            it = items[0]
            print(f"  [{key}] {it.item_id}")
            print(f"      出典: {it.attribution}")
            print(f"      本文: {it.text[:70].replace(chr(10), ' / ')}…")

    # 正解の前提条件: 本文は正規化済み・長さは範囲内
    for key, items in per_source.items():
        for it in items:
            assert it.text == normalize(it.text), f"{it.item_id} が未正規化"
            assert it.license == LICENSES[key]["license"]
            assert _ja_chars(it.text) >= MIN_JA_CHARS

    # --- 3. キャッシュ経路の確認 (2 回目はネットワークに触らない) --------
    print("-" * 72)
    for key in LICENSES:
        p = cache_path(key, cache_dir=CACHE)
        n_lines = len(read_cache(key, cache_dir=CACHE))
        exists = "あり" if os.path.exists(p) else "なし"
        print(f"cache {p:<30} 実体{exists}  {n_lines}件")

    mid = net_stats()
    corpus1 = load_base_corpus(9, seed=3, cache_dir=CACHE)
    end = net_stats()
    print(f"load_base_corpus(9) -> {len(corpus1)}件 / "
          f"追加 HTTP リクエスト {end['requests'] - mid['requests']}")
    from collections import Counter

    print("  内訳:", dict(Counter(i.source for i in corpus1)))

    corpus2 = load_base_corpus(9, seed=3, cache_dir=CACHE)
    same = [a.item_id for a in corpus1] == [b.item_id for b in corpus2]
    print(f"  同一 seed で再現一致: {same}")
    assert same, "seed 固定でも結果が揺れている"

    os.environ[OFFLINE_ENV] = "1"
    off_before = net_stats()
    corpus3 = load_base_corpus(9, seed=3, cache_dir=CACHE)
    off_after = net_stats()
    print(f"  SUMI_OFFLINE=1 -> {len(corpus3)}件 / "
          f"HTTP リクエスト {off_after['requests'] - off_before['requests']} "
          f"(0 ならキャッシュのみで動作)")
    assert off_after["requests"] == off_before["requests"], "オフラインで通信している"
    assert [a.item_id for a in corpus1] == [c.item_id for c in corpus3], \
        "オフライン再実行で内容が変わった"
    os.environ.pop(OFFLINE_ENV, None)

    # --- 4. ライセンス台帳 -----------------------------------------------
    print("-" * 72)
    for row in license_table():
        print(f"  {row['source']:<14} {row['license']:<34} 継承{row['継承条件']}")

    # --- 5. 未知ソース名は明示的に落ちる ---------------------------------
    try:
        load_base_corpus(1, sources=("gutenberg",), cache_dir=CACHE)
    except ValueError as exc:
        print(f"  未知ソースは ValueError: {exc}"[:100])
    else:  # pragma: no cover
        print("  未知ソースが素通りした", file=sys.stderr)
        raise SystemExit(1)

    print("=" * 72)
    print("OK: sumi.corpus 自己テスト完了")
