"""Sumi core datatypes: the contract every other module is written against.

Claim: 全主張の土台 (検出率 / 低誤検出 / CPU速度 / 可逆性 / 較正)。
スパンを「文字オフセット」で一貫して表現することが、規則層・モデル層・
較正・可逆マスキングの突き合わせを可能にしている。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable, Sequence


class PIIType(str, Enum):
    """検出対象の個人情報種別。

    Claim: 検出率 — 種別ごとの検出率を報告するための基本単位。
    値は BIO ラベル名およびマスク置換子 (``<NAME_1>`` 等) の接頭辞として使う。
    """

    NAME = "NAME"                  # 氏名 (姓・名・姓名)
    ADDRESS = "ADDRESS"            # 住所 (都道府県〜番地)
    PHONE = "PHONE"                # 電話番号 (固定・携帯・フリーダイヤル)
    EMAIL = "EMAIL"                # メールアドレス
    DOB = "DOB"                    # 生年月日 (西暦・和暦)
    BANK_ACCOUNT = "BANK_ACCOUNT"  # 金融機関口座 (銀行コード/支店/口座番号)
    CREDIT_CARD = "CREDIT_CARD"    # クレジットカード番号様式
    MYNUMBER = "MYNUMBER"          # マイナンバー様式の12桁
    MEMBER_ID = "MEMBER_ID"        # 各種会員番号・顧客番号
    POSTAL_CODE = "POSTAL_CODE"    # 郵便番号

    @property
    def ja(self) -> str:
        """種別の日本語表示名を返す。

        Claim: 検出率 — 図表・UI で種別別の検出率を日本語表示するため。
        """
        return _JA_NAMES[self.value]

    @property
    def en(self) -> str:
        """種別の英語表示名を返す。

        Claim: 検出率 — 公開する CLI・UI・ドキュメントは英語で提示するため。
        """
        return _EN_NAMES[self.value]


_EN_NAMES = {
    "NAME": "Name",
    "ADDRESS": "Address",
    "PHONE": "Phone",
    "EMAIL": "Email",
    "DOB": "Birth date",
    "BANK_ACCOUNT": "Bank account",
    "CREDIT_CARD": "Card number",
    "MYNUMBER": "My Number",
    "MEMBER_ID": "Member ID",
    "POSTAL_CODE": "Postal code",
}

_JA_NAMES = {
    "NAME": "氏名",
    "ADDRESS": "住所",
    "PHONE": "電話番号",
    "EMAIL": "メールアドレス",
    "DOB": "生年月日",
    "BANK_ACCOUNT": "金融口座",
    "CREDIT_CARD": "カード番号",
    "MYNUMBER": "マイナンバー様式",
    "MEMBER_ID": "会員番号",
    "POSTAL_CODE": "郵便番号",
}

#: 形式が確定していて規則層が高精度に拾える種別 (RuleLayer が優先される)。
RULE_DETERMINISTIC: frozenset[PIIType] = frozenset(
    {
        PIIType.EMAIL,
        PIIType.PHONE,
        PIIType.POSTAL_CODE,
        PIIType.BANK_ACCOUNT,
        PIIType.CREDIT_CARD,
        PIIType.MYNUMBER,
        PIIType.MEMBER_ID,
    }
)

#: 文脈判断が要る種別 (モデル層が主役)。
MODEL_DRIVEN: frozenset[PIIType] = frozenset(
    {PIIType.NAME, PIIType.ADDRESS, PIIType.DOB}
)

ALL_TYPES: tuple[PIIType, ...] = tuple(PIIType)


class Source(str, Enum):
    """スパンの出所。統合時の優先順位判断と、評価時の内訳集計に使う。

    Claim: 低誤検出 — 「どの層が出したスパンか」を残すことで、
    誤検出の責任を層ごとに切り分けられる。
    """

    MODEL = "model"
    RULE = "rule"
    MERGED = "merged"
    GOLD = "gold"
    BASELINE = "baseline"


@dataclass(frozen=True, slots=True)
class Span:
    """テキスト中の1件の検出結果 (または正解)。

    オフセットは **Python の文字インデックス** (UTF-32 相当、NFKC 正規化後の
    本文に対する半開区間 ``[start, end)``)。バイト長ではない。

    Claim: 検出率 / 低誤検出 / 可逆性 — 正解スパンと予測スパンを同じ座標系で
    比較できるため検出率・誤検出率が定義でき、同じ座標でマスクと復元ができる。

    Attributes:
        start: 開始文字位置 (含む)。
        end: 終了文字位置 (含まない)。
        label: PII 種別。
        text: 該当文字列 (照合とデバッグ用。原文から切り出した値と一致すべき)。
        score: 0..1 の確信度。較正後は「そのスパンが真である確率」の推定値。
        source: 出所 (model / rule / merged / gold / baseline)。
        meta: 付随情報 (``checksum_valid``、``rule_id``、``negative_kind`` など)。
    """

    start: int
    end: int
    label: PIIType
    text: str = ""
    score: float = 1.0
    source: Source = Source.MODEL
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span offsets: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        """スパンの文字数。

        Claim: 検出率 — IoU と境界ずれ量の計算に用いる基本量。
        """
        return self.end - self.start

    def key(self) -> tuple[int, int, str]:
        """厳密一致比較用のキー (start, end, label)。

        Claim: 検出率 — 厳密一致 (exact match) での検出率算定に用いる。
        """
        return (self.start, self.end, self.label.value)

    def overlaps(self, other: "Span") -> bool:
        """文字区間が1文字でも重なるか。

        Claim: 低誤検出 — 部分一致 (partial match) 判定と、
        規則層とモデル層の競合検出に用いる。
        """
        return self.start < other.end and other.start < self.end

    def iou(self, other: "Span") -> float:
        """文字単位の Jaccard 係数 (境界のゆれの厳しさを測る)。

        Claim: 低誤検出 — 「敬称を巻き込んだ」等の境界ずれを、
        検出/誤検出の二値ではなく連続量で評価するため。
        """
        inter = max(0, min(self.end, other.end) - max(self.start, other.start))
        if inter == 0:
            return 0.0
        union = self.length + other.length - inter
        return inter / union

    def with_(self, **kw: Any) -> "Span":
        """一部フィールドを差し替えた複製を返す (frozen dataclass 用)。

        Claim: 可逆性 — スパンを不変に保ったまま統合・較正できるので、
        マスク時と復元時で座標がずれない。
        """
        d = asdict(self)
        d["label"] = self.label
        d["source"] = self.source
        d.update(kw)
        return Span(**d)

    def slice_of(self, text: str) -> str:
        """原文から該当区間を切り出す。

        Claim: 可逆性 — マスク前後で同じ座標が同じ文字列を指すことの確認に使う。
        """
        return text[self.start : self.end]

    def to_dict(self) -> dict[str, Any]:
        """JSON 化可能な辞書へ変換する。

        Claim: 検出率 — データセット公開と評価結果の永続化に用いる共通表現。
        """
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label.value,
            "text": self.text,
            "score": round(float(self.score), 6),
            "source": self.source.value,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Span":
        """``to_dict`` の逆変換。

        Claim: 検出率 — 公開データセットを読み戻しても評価が再現することを保証する。
        """
        return Span(
            start=int(d["start"]),
            end=int(d["end"]),
            label=PIIType(d["label"]),
            text=d.get("text", ""),
            score=float(d.get("score", 1.0)),
            source=Source(d.get("source", "model")),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class Document:
    """1件の評価/学習単位。本文と正解スパン、由来メタデータを持つ。

    Claim: 検出率 / 低誤検出 — 合成であること・出典ライセンス・
    否定例かどうかをレコードに保持し、評価を層別に集計できるようにする。

    Attributes:
        text: NFKC 正規化済みの本文。
        spans: 正解 PII スパン (合成挿入時に構成的に記録したもの)。
        doc_id: 一意 ID。
        subset: ``"train" | "validation" | "test" | "negatives"``。
        genre: ``"email" | "minutes" | "application" | "inquiry" | "wiki" | ...``
        source_license: 土台テキストのライセンス表記 (例 ``"CC BY-SA 4.0"``)。
        source_ref: 土台テキストの出典 (URL や作品ID)。
        negative_kinds: 混入させた hard negative の種別ラベル。
        meta: その他。
    """

    text: str
    spans: list[Span] = field(default_factory=list)
    doc_id: str = ""
    subset: str = "train"
    genre: str = "synthetic"
    source_license: str = "synthetic (CC0-1.0)"
    source_ref: str = ""
    negative_kinds: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def sorted_spans(self) -> list[Span]:
        """スパンを開始位置昇順で返す。

        Claim: 可逆性 — マスク処理は順序に依存するため、
        常に決定論的な順序を与える。
        """
        return sorted(self.spans, key=lambda s: (s.start, s.end))

    def validate(self) -> None:
        """正解スパンが本文と整合しているか検証する (構成的正解の自己点検)。

        Claim: 検出率 — 正解が壊れていれば検出率の数値自体が無意味になるため、
        データ生成時点で ``text[start:end] == span.text`` を強制する。
        """
        n = len(self.text)
        prev_end = -1
        for s in self.sorted_spans():
            if s.end > n:
                raise ValueError(f"{self.doc_id}: span {s.key()} exceeds text length {n}")
            if s.text and self.text[s.start : s.end] != s.text:
                raise ValueError(
                    f"{self.doc_id}: span text mismatch at {s.key()}: "
                    f"{self.text[s.start:s.end]!r} != {s.text!r}"
                )
            if s.start < prev_end:
                raise ValueError(f"{self.doc_id}: overlapping gold spans near {s.key()}")
            prev_end = s.end

    def to_dict(self) -> dict[str, Any]:
        """JSON 化可能な辞書へ変換する (HF Dataset の1行に対応)。

        Claim: 検出率 — 合成であることとライセンスを行ごとに保持したまま公開する。
        """
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "spans": [s.to_dict() for s in self.sorted_spans()],
            "subset": self.subset,
            "genre": self.genre,
            "source_license": self.source_license,
            "source_ref": self.source_ref,
            "negative_kinds": self.negative_kinds,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Document":
        """``to_dict`` の逆変換。

        Claim: 検出率 — 公開データセットからの評価再現性を保証する。
        """
        return Document(
            text=d["text"],
            spans=[Span.from_dict(x) for x in d.get("spans", [])],
            doc_id=d.get("doc_id", ""),
            subset=d.get("subset", "train"),
            genre=d.get("genre", "synthetic"),
            source_license=d.get("source_license", "synthetic (CC0-1.0)"),
            source_ref=d.get("source_ref", ""),
            negative_kinds=list(d.get("negative_kinds") or []),
            meta=dict(d.get("meta") or {}),
        )


def normalize(text: str) -> str:
    """本文の正規化 (NFKC + 改行統一)。全モジュールで **必ず** 最初に通す。

    Claim: 検出率 / 低誤検出 — 全角英数字・半角カナ・異体ハイフンの揺れを
    吸収し、規則層の正規表現とモデルの語彙を安定させる。
    正規化はスパン座標を変えうるため、**挿入前の土台テキストに対して1度だけ**
    適用し、以後は正規化済み文字列だけを扱う。
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # NFKC が潰しきらない各種ダッシュ類を ASCII ハイフンに寄せる。
    # カタカナ長音符 U+30FC "ー" は **絶対に含めない** (コーヒー -> コ-ヒ- を防ぐ)。
    for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFF0D\u30FB"[:-1]:
        t = t.replace(ch, "-")
    return t


def bio_labels(types: Sequence[PIIType] = ALL_TYPES) -> list[str]:
    """BIO ラベル一覧を決定論的な順序で返す (``O`` が index 0)。

    Claim: 検出率 — 学習・推論・ONNX 書き出しでラベル順序が一致することを保証し、
    モデル差し替え時のラベルずれ由来の検出率低下を防ぐ。
    """
    out = ["O"]
    for t in types:
        out.append(f"B-{t.value}")
        out.append(f"I-{t.value}")
    return out


def spans_to_bio(
    spans: Iterable[Span], offsets: Sequence[tuple[int, int]], label2id: dict[str, int]
) -> list[int]:
    """文字スパンを、トークン offset 列に対する BIO ラベル ID 列へ変換する。

    Claim: 検出率 — サブワード境界と PII 境界のずれを「重なれば内側」の規則で
    吸収し、境界不一致による学習信号の欠落を最小化する。
    特殊トークン (offset が ``(0, 0)`` などの零幅) は ``-100`` (loss 無視)。
    """
    out: list[int] = []
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    for (a, b) in offsets:
        if b <= a:
            out.append(-100)
            continue
        lab = "O"
        for s in ordered:
            if a < s.end and s.start < b:
                lab = f"B-{s.label.value}" if a <= s.start else f"I-{s.label.value}"
                break
        out.append(label2id[lab])
    return out
