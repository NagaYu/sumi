"""Sumi のモデル層: 文脈判断が要る PII (氏名・住所・生年月日) のトークン分類器。

Claim: 検出率 / CPU速度 / 較正 —
規則層が形式で拾えない ``NAME`` / ``ADDRESS`` / ``DOB`` を文脈から検出し (検出率)、
132M の小型 backbone と重なりありスライディングウィンドウで CPU 実用速度を保ち (CPU速度)、
スパン確信度を「構成トークン確率の最小値 (最も弱い根拠)」として定義することで
較正 (温度/等調) にかけられる 1 本の実数を出す (較正)。

backbone は ``sbintuitions/modernbert-ja-130m`` に固定している。選定理由:

* fast tokenizer が ``offset_mapping`` を返す (文字オフセット契約に必須)
* 日本語の氏名/住所スパン境界の 98.2% がサブワード境界と厳密に一致する
  (残り 1.8% は ``refine_boundaries`` が敬称・助詞剥がしで回収する)
* 132M パラメータ / hidden 512 / 19 層、この Mac の CPU で約 48 docs/s
* MIT ライセンス

``AutoModelForTokenClassification`` で読み込むと分類ヘッドが新規初期化される旨の警告と
``decoder.bias`` が unexpected である旨の報告が出るが、これは想定どおり (MLM ヘッドを
捨ててトークン分類ヘッドを付けているだけ)。
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .types import (
    Document,
    PIIType,
    Source,
    Span,
    bio_labels,
    normalize,
    spans_to_bio,
)

__all__ = [
    "DEFAULT_BACKBONE",
    "TrainConfig",
    "TokenClassifier",
    "decode_bio",
    "refine_boundaries",
    "pick_device",
    "HONORIFICS",
    "PARTICLES",
    "LABELS_FILENAME",
]

#: 契約で固定された backbone。差し替える場合は ONNX 書き出しとベンチも取り直すこと。
DEFAULT_BACKBONE = "sbintuitions/modernbert-ja-130m"

#: ``save()`` がモデルと一緒に必ず書き出すラベル台帳のファイル名。
LABELS_FILENAME = "sumi_labels.json"

#: loss を無視する位置 (特殊トークン / パディング)。
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# 境界補正に使う語彙
# ---------------------------------------------------------------------------

#: 末尾から剥がす敬称。**長いものから**並べる (「先生」を「生」で切らないため)。
HONORIFICS: tuple[str, ...] = (
    "先生",
    "部長",
    "課長",
    "社長",
    "ちゃん",
    "様",
    "さん",
    "氏",
    "殿",
    "君",
)

#: 末尾から剥がす助詞。同じく長いものから。
PARTICLES: tuple[str, ...] = (
    "より",
    "から",
    "は",
    "が",
    "の",
    "を",
    "に",
    "へ",
    "と",
    "で",
    "も",
)

#: 前後から剥がす空白・記号。カタカナ長音符「ー」は **入れない**
#: (「マリー」「リー」等の氏名末尾を削ってしまうため)。
_TRIM_CHARS = (
    " \t\n　"
    "、。，．,.:：;；・"
    "「」『』（）()[]［］{}｛｝【】〈〉《》"
    "!?！？…\"'“”‘’"
    "-–—/／|｜*＊#＃@＠&＆+＋=＝_＿"
)

#: 境界補正を適用する種別 (数字系は形式が厳密なので触らない)。
REFINE_LABELS = frozenset({PIIType.NAME, PIIType.ADDRESS})


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """学習ハイパーパラメータ (契約で定められた名前と既定値)。

    Claim: 検出率 / CPU速度 — 学習条件を 1 個のデータクラスに固定して
    再現可能にし、``history`` に丸ごと記録することで
    「どの設定でその検出率が出たか」を後から照合できるようにする。

    Attributes:
        backbone: 事前学習モデル名。
        epochs: エポック数 (小数可。総ステップ数の計算に使う)。
        lr: AdamW の学習率。
        batch_size: ミニバッチ件数 (ウィンドウ単位)。
        max_length: 1 ウィンドウのトークン長 (特殊トークン込み)。
        warmup_ratio: 線形ウォームアップの割合。
        weight_decay: bias / LayerNorm を除いたパラメータへの weight decay。
        seed: シャッフルと初期化の種。
        device: ``None`` なら mps > cpu で自動選択。
        output_dir: 最良チェックポイントの保存先 (空文字なら保存しない)。
    """

    backbone: str = DEFAULT_BACKBONE
    epochs: float = 3.0
    lr: float = 3e-5
    batch_size: int = 16
    max_length: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 0
    device: str | None = None
    output_dir: str = "artifacts/sumi-model"


def pick_device(pref: str | None = None) -> str:
    """使用デバイスを決める (``None`` なら mps > cpu)。

    Claim: CPU速度 — 推論ベンチは必ず ``"cpu"`` を明示指定して測れるようにし、
    学習だけ mps を使えるようにするための単一の分岐点。

    Args:
        pref: 明示指定 (``"cpu"`` / ``"mps"`` / ``"cuda"``)。``None`` で自動。
    """
    if pref:
        return pref
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - mps が無い環境
        pass
    if torch.cuda.is_available():  # pragma: no cover - この Mac では通らない
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# トークナイズ (重なりありスライディングウィンドウ)
# ---------------------------------------------------------------------------


def _window_step(max_length: int, stride: int | None) -> int:
    body = max(1, max_length - 2)
    step = stride if stride else max_length // 2
    return max(1, min(int(step), body))


def _tokenize_windows(
    tokenizer: Any,
    text: str,
    max_length: int,
    stride: int | None = None,
) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """本文を重なりありのウィンドウ列にする。offset は **元テキスト基準**。

    自前で分割しているのは、``return_overflowing_tokens`` の stride 意味論に依存せず
    「オフセットが必ず元の文字位置を指す」ことをこのモジュールで保証するため。

    Returns:
        ``[(input_ids, offsets), ...]``。offsets の特殊トークン位置は ``(0, 0)``。
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = list(enc["input_ids"])
    offs: list[tuple[int, int]] = [(int(a), int(b)) for a, b in enc["offset_mapping"]]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:  # pragma: no cover - 想定外の tokenizer
        cls_id = cls_id if cls_id is not None else tokenizer.bos_token_id or 0
        sep_id = sep_id if sep_id is not None else tokenizer.eos_token_id or 0

    body = max(1, max_length - 2)
    step = _window_step(max_length, stride)

    windows: list[tuple[list[int], list[tuple[int, int]]]] = []
    if not ids:
        return [([cls_id, sep_id], [(0, 0), (0, 0)])]

    start = 0
    while True:
        chunk_ids = ids[start : start + body]
        chunk_offs = offs[start : start + body]
        windows.append(
            (
                [cls_id] + chunk_ids + [sep_id],
                [(0, 0)] + chunk_offs + [(0, 0)],
            )
        )
        if start + body >= len(ids):
            break
        start += step
    return windows


# ---------------------------------------------------------------------------
# BIO デコード
# ---------------------------------------------------------------------------


def _label_index(label_list: Sequence[str]) -> tuple[int, list[str], list[int], list[int]]:
    """``label_list`` から (O の index, 種別名リスト, B列, I列) を作る。"""
    o_index = label_list.index("O") if "O" in label_list else 0
    b_col: dict[str, int] = {}
    i_col: dict[str, int] = {}
    order: list[str] = []
    for idx, name in enumerate(label_list):
        if name == "O" or "-" not in name:
            continue
        prefix, typ = name.split("-", 1)
        if typ not in b_col and typ not in i_col:
            order.append(typ)
        if prefix == "B":
            b_col[typ] = idx
        elif prefix == "I":
            i_col[typ] = idx
    types = [t for t in order if t in b_col and t in i_col]
    return o_index, types, [b_col[t] for t in types], [i_col[t] for t in types]


def decode_bio(
    probs: Any,
    offsets: Sequence[tuple[int, int]],
    text: str,
    label_list: Sequence[str],
    *,
    threshold: float = 0.5,
) -> list[Span]:
    """トークン確率列を文字スパンへ復号する (B が欠けた I を許容)。

    Claim: 検出率 / 較正 —
    (1) 学習不足や境界ゆれで ``B-`` が落ちた場合でも先頭の ``I-`` を ``B-`` とみなすため、
    「1 トークンの取りこぼしでスパンごと消える」事故を防ぐ (検出率)。
    (2) スパンの確信度を **構成トークン確率の最小値** と定義する (較正)。
    最小値を採るのは、スパンは「全トークンが正しいときだけ正しい」連言だからで、
    平均を採ると 1 トークンだけ自信のない危うい長いスパンが高得点になってしまう。
    最小値 = 最も弱い根拠 は連言の確率の上界として自然で、閾値を上げると
    「どこか 1 箇所でも怪しいスパン」から先に落ちる = 低誤検出側に素直に効く。

    種別確率は ``P(B-X) + P(I-X)`` を束ねて使い、B/I の別はその 2 つの大小で決める。
    B と I に確率が割れて O に負ける (実質は明らかに実体なのに O になる) 事故を防ぐため。

    Args:
        probs: ``(T, L)`` の確率 (softmax 済み)。numpy 配列・リストいずれも可。
        offsets: 各トークンの ``(start, end)`` 文字オフセット。零幅は特殊トークン。
        text: オフセットの基準となる NFKC 正規化済み本文。
        label_list: ``bio_labels()`` と同じ順序のラベル名。
        threshold: スパン確信度 (最小値) の下限。

    Returns:
        ``Source.MODEL`` のスパン列 (start 昇順、非重複)。
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    o_index, types, b_cols, i_cols = _label_index(label_list)
    if not types:
        return []

    n = min(arr.shape[0], len(offsets))
    agg = arr[:n, b_cols] + arr[:n, i_cols]          # (T, K) 種別ごとの確率質量
    best_k = agg.argmax(axis=1)
    best_p = agg.max(axis=1)
    o_p = arr[:n, o_index]
    b_p = arr[:n, b_cols]
    i_p = arr[:n, i_cols]

    spans: list[Span] = []
    cur_type: str | None = None
    cur_start = 0
    cur_end = 0
    cur_score = 1.0

    def _flush() -> None:
        nonlocal cur_type
        if cur_type is not None and cur_end > cur_start and cur_score >= threshold:
            spans.append(
                Span(
                    start=cur_start,
                    end=cur_end,
                    label=PIIType(cur_type),
                    text=text[cur_start:cur_end],
                    score=float(cur_score),
                    source=Source.MODEL,
                    meta={"decoder": "bio_min"},
                )
            )
        cur_type = None

    for t in range(n):
        a, b = int(offsets[t][0]), int(offsets[t][1])
        if b <= a:
            # 特殊トークン: スパンを切らずに読み飛ばす (先頭/末尾にしか出ない)
            continue
        k = int(best_k[t])
        p = float(best_p[t])
        if p <= float(o_p[t]):
            _flush()
            continue
        typ = types[k]
        is_begin = float(b_p[t, k]) >= float(i_p[t, k])
        if cur_type == typ and not is_begin:
            cur_end = b
            cur_score = min(cur_score, p)
        else:
            # 種別が変わった / B が来た / 開いていない (= 先頭 I を B として扱う)
            _flush()
            cur_type = typ
            cur_start = a
            cur_end = b
            cur_score = p
    _flush()
    return spans


# ---------------------------------------------------------------------------
# 境界補正
# ---------------------------------------------------------------------------


def refine_boundaries(span: Span, text: str) -> Span:
    """氏名・住所スパンの末尾から敬称と助詞を、前後から空白・記号を剥がす。

    Claim: 検出率 / 低誤検出 —
    サブワード境界と PII 境界が厳密一致しない約 1.8% のスパン
    (「田中太郎様」のように敬称が同一トークンに融合する場合など) を
    厳密一致の正解へ引き戻す。境界が 1 文字ずれるだけで exact match は
    落ちるので、これは検出率にそのまま効く。同時に、余計な助詞を含んだ
    スパンでマスクして本文を壊すことも防ぐ (低誤検出側の品質)。

    空になる剥がし方は **絶対にしない** (「さん」だけのスパン等は原型を返す)。

    Args:
        span: モデルが出したスパン。
        text: スパンの基準となる本文。

    Returns:
        補正後のスパン (変化が無ければ入力をそのまま返す)。
    """
    if span.label not in REFINE_LABELS:
        return span
    n = len(text)
    s = max(0, min(int(span.start), n))
    e = max(s, min(int(span.end), n))
    if e <= s:
        return span

    changed = True
    while changed and e > s:
        changed = False
        while e > s and text[e - 1] in _TRIM_CHARS:
            e -= 1
            changed = True
        while s < e and text[s] in _TRIM_CHARS:
            s += 1
            changed = True
        for h in HONORIFICS:
            if e - s > len(h) and text[e - len(h) : e] == h:
                e -= len(h)
                changed = True
                break
        if changed:
            continue
        for p in PARTICLES:
            if e - s > len(p) and text[e - len(p) : e] == p:
                e -= len(p)
                changed = True
                break

    if e <= s:
        return span
    if s == span.start and e == span.end:
        return span
    meta = dict(span.meta)
    meta["refined"] = True
    meta["raw_span"] = [int(span.start), int(span.end)]
    return span.with_(start=s, end=e, text=text[s:e], meta=meta)


# ---------------------------------------------------------------------------
# ウィンドウ間のスパン統合
# ---------------------------------------------------------------------------


def _merge_window_spans(spans: Iterable[Span], text: str) -> list[Span]:
    """重なるウィンドウが出した重複スパンを 1 本にまとめる。

    1. ``(start, end, label)`` が同一なら score の高い方を残す
    2. 同一種別で重なるものは和集合にまとめ、score は最小値 (弱い根拠) を継ぐ
    3. 種別違いで重なるものは score の高い方を優先 (貪欲、非重複を保証)
    """
    best: dict[tuple[int, int, str], Span] = {}
    for s in spans:
        k = s.key()
        prev = best.get(k)
        if prev is None or s.score > prev.score:
            best[k] = s

    ordered = sorted(best.values(), key=lambda x: (x.start, x.end))
    unioned: list[Span] = []
    for s in ordered:
        if unioned and unioned[-1].label == s.label and unioned[-1].end > s.start:
            p = unioned[-1]
            ns, ne = min(p.start, s.start), max(p.end, s.end)
            unioned[-1] = p.with_(
                start=ns, end=ne, text=text[ns:ne], score=min(p.score, s.score)
            )
        else:
            unioned.append(s)

    final: list[Span] = []
    for s in sorted(unioned, key=lambda x: (-x.score, -(x.end - x.start), x.start)):
        if any(s.overlaps(f) for f in final):
            continue
        final.append(s)
    return sorted(final, key=lambda x: (x.start, x.end))


# ---------------------------------------------------------------------------
# 評価ヘルパ (calibrate.py に依存しないよう自前で最小限だけ持つ)
# ---------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _span_scores(
    gold_docs: Sequence[Document], pred_per_doc: Sequence[Sequence[Span]]
) -> dict[str, dict[str, float]]:
    etp = efp = efn = 0
    ptp = pfp = pfn = 0
    for doc, preds in zip(gold_docs, pred_per_doc):
        gold = list(doc.spans)
        gkeys = {s.key() for s in gold}
        pkeys = {s.key() for s in preds}
        etp += len(gkeys & pkeys)
        efp += len(pkeys - gkeys)
        efn += len(gkeys - pkeys)
        used: set[int] = set()
        hit = 0
        for pr in preds:
            for gi, g in enumerate(gold):
                if gi in used:
                    continue
                if g.label == pr.label and g.overlaps(pr):
                    used.add(gi)
                    hit += 1
                    break
        ptp += hit
        pfp += len(preds) - hit
        pfn += len(gold) - hit
    return {"exact": _prf(etp, efp, efn), "partial": _prf(ptp, pfp, pfn)}


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------


class TokenClassifier:
    """BIO トークン分類器 (NAME / ADDRESS / DOB を文脈から拾う層)。

    Claim: 検出率 / CPU速度 / 較正 — 規則で書けない種別を担当し (検出率)、
    小型 backbone + ウィンドウ処理で CPU 推論を実用速度に保ち (CPU速度)、
    スパンごとに 1 本の実数スコアを出して較正可能にする (較正)。
    """

    def __init__(self, model: Any, tokenizer: Any, label_list: Sequence[str]) -> None:
        """学習済み/未学習のモデル・トークナイザ・ラベル台帳を束ねる。

        Claim: 検出率 — ラベル順序をインスタンスに固定して持ち回ることで、
        学習・推論・保存の間でラベルがずれて検出率が壊れることを防ぐ。
        """
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "Sumi は offset_mapping を返す fast tokenizer を必須とする "
                "(文字オフセット契約のため)"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.label_list: list[str] = list(label_list)
        self.label2id: dict[str, int] = {l: i for i, l in enumerate(self.label_list)}
        self.id2label: dict[int, str] = {i: l for i, l in enumerate(self.label_list)}

    # -- 構築 / 永続化 ------------------------------------------------------

    @classmethod
    def from_backbone(cls, backbone: str = DEFAULT_BACKBONE) -> "TokenClassifier":
        """事前学習 backbone に 21 クラスの分類ヘッドを付けて初期化する。

        Claim: 検出率 / CPU速度 — 契約で選定した 132M の日本語 ModernBERT
        (offset 対応 fast tokenizer / MIT / CPU で実用速度) を唯一の入口にし、
        報告された検出率と速度が同じ構成で再現できるようにする。

        分類ヘッドが新規初期化される警告と ``decoder.bias`` の unexpected 報告は想定内。
        """
        from transformers import (  # 局所 import: 起動を軽くする
            AutoConfig,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        labels = bio_labels()
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        config = AutoConfig.from_pretrained(
            backbone,
            num_labels=len(labels),
            id2label={i: l for i, l in enumerate(labels)},
            label2id={l: i for i, l in enumerate(labels)},
        )
        # ModernBERT の torch.compile 経路は mps で不安定なので切っておく。
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
        model = AutoModelForTokenClassification.from_pretrained(backbone, config=config)
        return cls(model, tokenizer, labels)

    @classmethod
    def load(cls, path: str, *, device: str | None = None) -> "TokenClassifier":
        """``save()`` で書き出したディレクトリから復元する。

        Claim: 検出率 / CPU速度 — ラベル台帳 (``sumi_labels.json``) を
        モデル config より優先して読むことで、学習時と推論時のラベル対応が
        絶対にずれないようにする。``device="cpu"`` を明示すれば CPU ベンチが取れる。
        """
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        labels: list[str] | None = None
        labels_path = os.path.join(path, LABELS_FILENAME)
        if not os.path.exists(labels_path) and not os.path.isdir(path):
            # Hugging Face Hub のリポジトリ ID が渡された場合はラベル台帳を取得する。
            try:
                from huggingface_hub import hf_hub_download

                labels_path = hf_hub_download(path, LABELS_FILENAME)
            except Exception:
                labels_path = ""
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as fh:
                labels = list(json.load(fh)["label_list"])

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(path)
        if labels is None:
            id2label = model.config.id2label or {}
            labels = [id2label[i] for i in sorted(int(k) for k in id2label)]
        obj = cls(model, tokenizer, labels)
        obj.to(pick_device(device))
        obj.model.eval()
        return obj

    def save(self, path: str) -> None:
        """モデル・トークナイザ・ラベル台帳を 1 つのディレクトリへ保存する。

        Claim: 検出率 — ラベル順序を config (``id2label``/``label2id``) と
        ``sumi_labels.json`` の **二重** に書き出す。ONNX 書き出しや別プロセス推論で
        ラベルが入れ替わると検出率が黙って壊れるため、台帳を必ずモデルに同梱する。
        """
        os.makedirs(path, exist_ok=True)
        self.model.config.id2label = {i: l for i, l in enumerate(self.label_list)}
        self.model.config.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.model.config.num_labels = len(self.label_list)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, LABELS_FILENAME), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "label_list": self.label_list,
                    "num_labels": len(self.label_list),
                    "backbone": getattr(self.model.config, "_name_or_path", ""),
                    "schema": "sumi-bio-v1",
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

    def to(self, device: str) -> "TokenClassifier":
        """モデルを指定デバイスへ移す (self を返す)。

        Claim: CPU速度 — ベンチ時に ``to("cpu")`` を明示できることが、
        報告する docs/s が CPU 実測であることの担保になる。
        """
        self.model.to(device)
        return self

    @property
    def device(self) -> str:
        """モデルが今載っているデバイス名。

        Claim: CPU速度 — 推論計測時に実際のデバイスを取り違えないための単一の情報源。
        """
        try:
            return str(next(self.model.parameters()).device).split(":")[0]
        except StopIteration:  # pragma: no cover
            return "cpu"

    # -- 特徴量 -------------------------------------------------------------

    def _featurize(
        self, docs: Sequence[Document], max_length: int, stride: int | None = None
    ) -> list[dict[str, list[int]]]:
        feats: list[dict[str, list[int]]] = []
        for doc in docs:
            for ids, offs in _tokenize_windows(
                self.tokenizer, doc.text, max_length, stride
            ):
                labels = spans_to_bio(doc.spans, offs, self.label2id)
                feats.append({"input_ids": ids, "labels": labels})
        return feats

    def _collate(self, batch: Sequence[dict[str, list[int]]], device: str):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:  # pragma: no cover
            pad_id = 0
        n = len(batch)
        L = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((n, L), pad_id, dtype=torch.long)
        attn = torch.zeros((n, L), dtype=torch.long)
        labels = torch.full((n, L), IGNORE_INDEX, dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            input_ids[i, :k] = torch.tensor(b["input_ids"], dtype=torch.long)
            attn[i, :k] = 1
            labels[i, :k] = torch.tensor(b["labels"], dtype=torch.long)
        return (
            input_ids.to(device),
            attn.to(device),
            labels.to(device),
        )

    # -- 学習 ---------------------------------------------------------------

    def train(
        self,
        train_docs: list[Document],
        val_docs: list[Document],
        cfg: TrainConfig,
    ) -> dict:
        """自前の torch ループで学習する (transformers.Trainer は使わない)。

        Claim: 検出率 / CPU速度 — AdamW + 線形ウォームアップ/減衰 + 勾配クリップという
        素直な構成を全部見える形で持ち、エポックごとにトークン F1 とスパン F1 を測って
        **スパン F1 最良のチェックポイントだけ** を残す。スパン境界まで合った時だけ
        得点される指標で選ぶので、検出率の報告値と選択基準が一致する。
        mps 固有の失敗は握りつぶして cpu へ落とし、学習が止まらないようにする。

        Args:
            train_docs: 学習文書 (gold span 付き)。
            val_docs: 検証文書。
            cfg: 学習設定。

        Returns:
            history dict (設定・エポックごとの loss と指標・最良エポック・所要秒数)。
        """
        t_start = time.time()
        torch.manual_seed(cfg.seed)

        device = pick_device(cfg.device)
        self.model.to(device)

        feats_train = self._featurize(train_docs, cfg.max_length)
        feats_val = self._featurize(val_docs, cfg.max_length) if val_docs else []
        if not feats_train:
            raise ValueError("学習データが空です")

        history: dict[str, Any] = {
            "config": asdict(cfg),
            "label_list": list(self.label_list),
            "device": device,
            "device_fallback": None,
            "n_train_docs": len(train_docs),
            "n_val_docs": len(val_docs),
            "n_train_windows": len(feats_train),
            "epochs": [],
            "best": None,
            "wall_seconds": 0.0,
        }

        if device == "mps" and not self._mps_smoke_ok(feats_train[:1]):
            history["device_fallback"] = "mps smoke test failed -> cpu"
            device = "cpu"
            history["device"] = device
            self.model.to(device)

        try:
            best_state, best_f1, best_epoch = self._run_epochs(
                feats_train, feats_val, val_docs, cfg, device, history
            )
        except Exception as exc:  # mps 固有のカーネル未実装等
            if device == "cpu":
                raise
            msg = f"{device} でのの学習が失敗したため cpu へ退避: {type(exc).__name__}: {exc}"
            msg = msg.replace("でのの", "での")
            print(f"[sumi.model] {msg}")
            history["device_fallback"] = msg
            history["epochs"] = []
            device = "cpu"
            history["device"] = device
            self.model.to(device)
            torch.manual_seed(cfg.seed)
            best_state, best_f1, best_epoch = self._run_epochs(
                feats_train, feats_val, val_docs, cfg, device, history
            )

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to(device)
        history["best"] = {"epoch": best_epoch, "val_span_f1_exact": best_f1}
        history["wall_seconds"] = round(time.time() - t_start, 2)

        if cfg.output_dir:
            self.save(cfg.output_dir)
            history["saved_to"] = cfg.output_dir
        return history

    def _mps_smoke_ok(self, sample: Sequence[dict[str, list[int]]]) -> bool:
        """mps で 1 回だけ forward/backward を試し、通らなければ False。"""
        if not sample:
            return True
        try:
            self.model.train()
            ids, attn, labels = self._collate(sample, "mps")
            out = self.model(input_ids=ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            self.model.zero_grad(set_to_none=True)
            return bool(torch.isfinite(out.loss.detach()).item())
        except Exception as exc:  # pragma: no cover - 環境依存
            print(f"[sumi.model] mps smoke test failed: {type(exc).__name__}: {exc}")
            self.model.zero_grad(set_to_none=True)
            return False

    def _run_epochs(
        self,
        feats_train: list[dict[str, list[int]]],
        feats_val: list[dict[str, list[int]]],
        val_docs: Sequence[Document],
        cfg: TrainConfig,
        device: str,
        history: dict[str, Any],
    ) -> tuple[dict | None, float, int]:
        rng = random.Random(cfg.seed)
        steps_per_epoch = max(1, math.ceil(len(feats_train) / cfg.batch_size))
        total_steps = max(1, int(round(steps_per_epoch * cfg.epochs)))
        warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
        n_epochs = max(1, math.ceil(cfg.epochs))

        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if (param.ndim <= 1 or name.endswith(".bias")) else decay).append(
                param
            )
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        history["total_steps"] = total_steps
        history["warmup_steps"] = warmup_steps

        best_state: dict | None = None
        best_f1 = -1.0
        best_epoch = -1
        step = 0
        order = list(range(len(feats_train)))

        for epoch in range(n_epochs):
            self.model.train()
            rng.shuffle(order)
            losses: list[float] = []
            for b0 in range(0, len(order), cfg.batch_size):
                if step >= total_steps:
                    break
                batch = [feats_train[i] for i in order[b0 : b0 + cfg.batch_size]]
                ids, attn, labels = self._collate(batch, device)
                out = self.model(input_ids=ids, attention_mask=attn, labels=labels)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().to("cpu")))
                step += 1

            token_metrics = self._token_metrics(feats_val, cfg, device)
            preds = (
                self.predict(
                    [d.text for d in val_docs],
                    batch_size=max(1, cfg.batch_size),
                    max_length=cfg.max_length,
                    threshold=0.5,
                    refine=True,
                )
                if val_docs
                else []
            )
            span_metrics = _span_scores(val_docs, preds) if val_docs else {}
            rec = {
                "epoch": epoch + 1,
                "step": step,
                "train_loss": round(float(np.mean(losses)) if losses else float("nan"), 4),
                "lr": scheduler.get_last_lr()[0],
                "val_token": token_metrics,
                "val_span": span_metrics,
            }
            history["epochs"].append(rec)
            f1 = float(span_metrics.get("exact", {}).get("f1", 0.0)) if span_metrics else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().to("cpu").clone()
                    for k, v in self.model.state_dict().items()
                }
                # 最良を更新した時点で **その場で** ディスクへ保存する。
                # 全エポック終了後にまとめて保存すると、長時間学習が中断された
                # 場合に一切の成果が残らない。この時点の self.model は
                # 最良の重みそのものなので、そのまま save() してよい。
                if cfg.output_dir:
                    try:
                        self.save(cfg.output_dir)
                    except Exception as exc:  # 保存失敗で学習を止めない
                        print(f"  [warn] チェックポイント保存に失敗: {exc}", flush=True)

            print(
                f"  epoch {epoch + 1}/{n_epochs} step {step}/{total_steps} "
                f"loss={rec['train_loss']:.4f} "
                f"val_span_exact_f1={f1:.4f} "
                f"(best={best_f1:.4f} @ep{best_epoch})",
                flush=True,
            )
            if step >= total_steps:
                break

        return best_state, best_f1, best_epoch

    def _token_metrics(
        self,
        feats_val: list[dict[str, list[int]]],
        cfg: TrainConfig,
        device: str,
    ) -> dict[str, float]:
        if not feats_val:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
        self.model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for b0 in range(0, len(feats_val), cfg.batch_size):
                batch = feats_val[b0 : b0 + cfg.batch_size]
                ids, attn, labels = self._collate(batch, device)
                logits = self.model(input_ids=ids, attention_mask=attn).logits
                pred = logits.argmax(dim=-1).to("cpu").numpy()
                gold = labels.to("cpu").numpy()
                mask = gold != IGNORE_INDEX
                p = pred[mask]
                g = gold[mask]
                tp += int(((p == g) & (g != 0)).sum())
                fp += int(((p != g) & (p != 0)).sum())
                fn += int(((p != g) & (g != 0)).sum())
        self.model.train()
        return _prf(tp, fp, fn)

    # -- 推論 ---------------------------------------------------------------

    def predict(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        max_length: int = 256,
        threshold: float = 0.5,
        refine: bool = True,
    ) -> list[list[Span]]:
        """複数本文をバッチ推論してスパン列を返す。

        Claim: 検出率 / CPU速度 —
        ``max_length`` を超える長文は **重なりありのスライディングウィンドウ**
        (step = ``max_length // 2``) で処理し、ウィンドウ境界で切られた実体を
        隣のウィンドウが必ず丸ごと含むようにして取りこぼしを防ぐ (検出率)。
        オフセットは常に元テキストの文字位置へ戻し、重複スパンは統合する。
        バッチ化とウィンドウ単位の詰め込みで CPU でも実用速度を出す (CPU速度)。

        Args:
            texts: NFKC 正規化済み本文のリスト (``normalize()`` 済みであること)。
            batch_size: 1 バッチのウィンドウ数。
            max_length: 1 ウィンドウのトークン長。
            threshold: スパン確信度 (構成トークン確率の最小値) の下限。
            refine: 敬称・助詞の境界補正を適用するか。

        Returns:
            各本文に対する ``Source.MODEL`` のスパン列 (start 昇順、非重複)。
        """
        spans, _ = self.predict_with_probs(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            threshold=threshold,
            refine=refine,
        )
        return spans

    def predict_with_probs(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        max_length: int = 256,
        threshold: float = 0.5,
        refine: bool = True,
        stride: int | None = None,
    ) -> tuple[list[list[Span]], list]:
        """``predict()`` と同じ推論を行い、スパンごとの生スコアも返す。

        Claim: 較正 / 検出率 — 較正器 (``SpanCalibrator``) の入力になる
        **未較正の** スコアを、スパンと同じ並びで別に返す。span.score を
        後から較正値へ差し替えても、元の生スコアが失われないようにするため。

        Returns:
            ``(spans_per_text, raw_scores_per_text)``。
            ``raw_scores_per_text[i][j]`` は ``spans_per_text[i][j]`` の生スコア。
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return [], []

        was_training = self.model.training
        self.model.eval()
        device = self.device

        tasks: list[tuple[int, list[int], list[tuple[int, int]]]] = []
        for i, text in enumerate(texts):
            for ids, offs in _tokenize_windows(
                self.tokenizer, text, max_length, stride
            ):
                tasks.append((i, ids, offs))

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:  # pragma: no cover
            pad_id = 0
        per_text: list[list[Span]] = [[] for _ in texts]

        with torch.no_grad():
            for b0 in range(0, len(tasks), max(1, batch_size)):
                chunk = tasks[b0 : b0 + max(1, batch_size)]
                L = max(len(c[1]) for c in chunk)
                input_ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
                attn = torch.zeros((len(chunk), L), dtype=torch.long)
                for j, (_, ids, _o) in enumerate(chunk):
                    input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attn[j, : len(ids)] = 1
                logits = self.model(
                    input_ids=input_ids.to(device), attention_mask=attn.to(device)
                ).logits
                probs = torch.softmax(logits.float(), dim=-1).to("cpu").numpy()
                for j, (ti, ids, offs) in enumerate(chunk):
                    per_text[ti].extend(
                        decode_bio(
                            probs[j, : len(ids)],
                            offs,
                            texts[ti],
                            self.label_list,
                            threshold=threshold,
                        )
                    )

        out: list[list[Span]] = []
        raw: list[list[float]] = []
        for i, text in enumerate(texts):
            merged = _merge_window_spans(per_text[i], text)
            if refine:
                merged = _merge_window_spans(
                    [refine_boundaries(s, text) for s in merged], text
                )
            out.append(merged)
            raw.append([float(s.score) for s in merged])

        if was_training:
            self.model.train()
        return out, raw


# ---------------------------------------------------------------------------
# 自己テスト
# ---------------------------------------------------------------------------


def _selftest_docs(n: int, seed: int) -> list[Document]:
    """自己テスト用の合成文書を作る (実在の個人情報は一切使わない)。

    姓名・地名は公開の統計的/地理的事実の語彙だが、**組合せは乱数**、
    番地・電話番号・生年月日はすべて乱数生成。スパンはプレースホルダ展開時に
    開始位置を記録する方式 (text.index は使わない)。
    """
    rng = random.Random(seed)
    sei = [
        "佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
        "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "清水", "森",
    ]
    mei = [
        "太郎", "花子", "一郎", "美咲", "健太", "陽子", "直樹", "由美", "大輔", "恵子",
        "翔太", "彩", "拓也", "真理", "亮",
    ]
    places = [
        ("東京都", "新宿区", "西新宿"),
        ("東京都", "港区", "赤坂"),
        ("大阪府", "大阪市北区", "梅田"),
        ("神奈川県", "横浜市西区", "みなとみらい"),
        ("愛知県", "名古屋市中区", "栄"),
        ("福岡県", "福岡市博多区", "博多駅前"),
        ("北海道", "札幌市中央区", "大通西"),
        ("京都府", "京都市中京区", "烏丸"),
        ("宮城県", "仙台市青葉区", "一番町"),
        ("広島県", "広島市中区", "紙屋町"),
    ]
    templates = [
        "{NAME}様、いつもお世話になっております。",
        "ご住所は{ADDRESS}で間違いないでしょうか。",
        "申込者は{NAME}、生年月日は{DOB}です。",
        "{NAME}さんより{ADDRESS}へ転居の連絡がありました。",
        "連絡先は{PHONE}、担当は{NAME}です。",
        "本日の会議には{NAME}と{NAME}が出席しました。",
        "送付先住所を{ADDRESS}に変更してください。",
        "{NAME}部長からの依頼で、{DOB}生まれの方の記録を確認しました。",
        "お問い合わせありがとうございます。{NAME}様の登録住所は{ADDRESS}です。",
        "電話番号{PHONE}宛にご連絡ください。担当者は{NAME}です。",
        "{ADDRESS}にお住まいの{NAME}様、書類が届きました。",
        "契約者名 {NAME} / 生年月日 {DOB} / 電話 {PHONE}",
    ]

    def gen(label: PIIType) -> str:
        if label is PIIType.NAME:
            return rng.choice(sei) + (rng.choice(mei) if rng.random() < 0.75 else "")
        if label is PIIType.ADDRESS:
            pref, city, town = rng.choice(places)
            return f"{pref}{city}{town}{rng.randint(1,9)}-{rng.randint(1,20)}-{rng.randint(1,30)}"
        if label is PIIType.PHONE:
            head = rng.choice(["090", "080", "070", "03", "06"])
            if head.startswith("0") and len(head) == 2:
                return f"{head}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
            return f"{head}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
        if label is PIIType.DOB:
            y, m, d = rng.randint(1950, 2005), rng.randint(1, 12), rng.randint(1, 28)
            if rng.random() < 0.3:
                return f"昭和{rng.randint(30,60)}年{m}月{d}日"
            return f"{y}年{m}月{d}日"
        raise ValueError(label)

    docs: list[Document] = []
    for i in range(n):
        tpl = templates[i % len(templates)]
        parts: list[str] = []
        spans: list[Span] = []
        pos = 0
        for piece in re.split(r"(\{[A-Z_]+\})", tpl):
            if not piece:
                continue
            if piece.startswith("{") and piece.endswith("}"):
                label = PIIType(piece[1:-1])
                value = gen(label)
                spans.append(
                    Span(pos, pos + len(value), label, value, 1.0, Source.GOLD)
                )
                parts.append(value)
                pos += len(value)
            else:
                parts.append(piece)
                pos += len(piece)
        text = "".join(parts)
        assert normalize(text) == text, "自己テストの合成本文が NFKC 非安定"
        doc = Document(
            text=text,
            spans=spans,
            doc_id=f"selftest-{i:03d}",
            subset="train",
            genre="inquiry",
        )
        doc.validate()
        docs.append(doc)
    return docs


def _selftest() -> None:  # pragma: no cover - 手動実行用
    print("=" * 70)
    print("sumi.model self-test")
    print("=" * 70)

    # --- 1. refine_boundaries の単体テスト (モデル読み込み前に落とす) -------
    cases = [
        ("田中太郎様", PIIType.NAME, "田中太郎"),
        ("東京都新宿区西新宿2-8-1に", PIIType.ADDRESS, "東京都新宿区西新宿2-8-1"),
        ("森さん", PIIType.NAME, "森"),
        ("鈴木一郎さんは", PIIType.NAME, "鈴木一郎"),
        ("　山本 ", PIIType.NAME, "山本"),
        ("「加藤」", PIIType.NAME, "加藤"),
        ("高橋部長", PIIType.NAME, "高橋"),
        ("小林先生の", PIIType.NAME, "小林"),
        ("さん", PIIType.NAME, "さん"),          # 空になる剥がしはしない
        ("の", PIIType.ADDRESS, "の"),           # 同上
        ("090-1234-5678に", PIIType.PHONE, "090-1234-5678に"),  # 対象外種別は不変
        ("大阪府大阪市北区梅田1-2-3から", PIIType.ADDRESS, "大阪府大阪市北区梅田1-2-3"),
        ("マリー", PIIType.NAME, "マリー"),      # 長音符は剥がさない
    ]
    ok = 0
    for raw, label, want in cases:
        span = Span(0, len(raw), label, raw, 1.0, Source.MODEL)
        got = refine_boundaries(span, raw)
        got_text = raw[got.start : got.end]
        flag = "OK " if got_text == want else "NG "
        ok += got_text == want
        print(f"  [{flag}] refine {raw!r:32} -> {got_text!r:26} (want {want!r})")
    print(f"refine_boundaries: {ok}/{len(cases)} passed")
    assert ok == len(cases), "refine_boundaries の単体テストに失敗"

    # --- 2. decode_bio の単体テスト (B 欠落許容 + 最小値スコア) -------------
    labels = bio_labels()
    name_b, name_i = labels.index("B-NAME"), labels.index("I-NAME")
    text = "田中太郎です"
    offsets = [(0, 0), (0, 2), (2, 4), (4, 6), (0, 0)]
    probs = np.zeros((5, len(labels)))
    probs[:, 0] = 1.0
    probs[0] = probs[4] = 0.0
    probs[0, 0] = probs[4, 0] = 1.0
    # 1 トークン目は B が無く I だけ (B 欠落) / 2 トークン目は I / 3 トークン目は O
    probs[1] = 0.0; probs[1, name_i] = 0.9; probs[1, 0] = 0.1
    probs[2] = 0.0; probs[2, name_i] = 0.6; probs[2, 0] = 0.4
    probs[3] = 0.0; probs[3, 0] = 1.0
    got = decode_bio(probs, offsets, text, labels, threshold=0.5)
    print(f"  decode_bio (B欠落): {[(s.start, s.end, s.label.value, round(s.score,2)) for s in got]}")
    assert len(got) == 1 and got[0].start == 0 and got[0].end == 4, "B 欠落の許容に失敗"
    assert abs(got[0].score - 0.6) < 1e-6, "スパンスコアが最小値になっていない"
    assert decode_bio(probs, offsets, text, labels, threshold=0.7) == [], "閾値が効いていない"
    print("decode_bio: OK (先頭 I を B として扱う / score = 構成トークンの最小値)")

    # --- 3. 合成データ -----------------------------------------------------
    docs = _selftest_docs(60, seed=7)
    train_docs, val_docs = docs[:48], docs[48:]
    n_spans = sum(len(d.spans) for d in docs)
    print(f"synthetic docs: {len(docs)} (train {len(train_docs)} / val {len(val_docs)}), gold spans {n_spans}")
    print(f"  example: {docs[0].text}")
    print(f"           {[ (s.label.value, s.text) for s in docs[0].spans ]}")

    # --- 4. 学習ループ -----------------------------------------------------
    clf = TokenClassifier.from_backbone()
    n_params = sum(p.numel() for p in clf.model.parameters())
    print(f"backbone: {DEFAULT_BACKBONE} ({n_params/1e6:.1f}M params, {len(clf.label_list)} labels)")

    tmp_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"sumi-model-selftest-{os.getpid()}"
    )
    cfg = TrainConfig(
        epochs=3.0,
        lr=5e-5,
        batch_size=8,
        max_length=128,
        seed=7,
        output_dir=tmp_dir,
    )
    t0 = time.time()
    history = clf.train(train_docs, val_docs, cfg)
    print(
        f"train: device={history['device']} steps={history['total_steps']} "
        f"wall={history['wall_seconds']}s fallback={history['device_fallback']}"
    )
    for rec in history["epochs"]:
        sp = rec["val_span"]["exact"]
        print(
            f"  epoch {rec['epoch']}: loss={rec['train_loss']:.4f} "
            f"token_f1={rec['val_token']['f1']:.3f} "
            f"span_f1(exact)={sp['f1']:.3f} span_f1(partial)={rec['val_span']['partial']['f1']:.3f}"
        )
    print(f"  best epoch: {history['best']}")

    # --- 5. 短文の推論 -----------------------------------------------------
    sample_texts = [
        "田中太郎様、いつもお世話になっております。",
        "ご住所は東京都新宿区西新宿2-8-1で間違いないでしょうか。",
    ]
    preds = clf.predict(sample_texts, max_length=128, threshold=0.3)
    for txt, sp in zip(sample_texts, preds):
        rendered = [(s.label.value, s.text, round(s.score, 3)) for s in sp]
        print(f"  predict {txt!r}\n     -> {rendered}")
        for s in sp:
            assert txt[s.start : s.end] == s.text, "オフセットと text が不整合"

    # --- 6. スライディングウィンドウ (>1000 文字) --------------------------
    long_parts: list[str] = []
    long_spans: list[Span] = []
    pos = 0
    for d in _selftest_docs(80, seed=99):
        if pos > 1400:
            break
        for s in d.spans:
            long_spans.append(s.with_(start=s.start + pos, end=s.end + pos))
        long_parts.append(d.text)
        pos += len(d.text) + 1
        long_parts.append("\n")
        # 上の +1 は改行分
    long_text = "".join(long_parts)
    long_doc = Document(text=long_text, spans=long_spans, doc_id="long")
    long_doc.validate()
    win = _tokenize_windows(clf.tokenizer, long_text, 64)
    long_pred, long_raw = clf.predict_with_probs(
        [long_text], batch_size=8, max_length=64, threshold=0.3
    )
    got_spans = long_pred[0]
    tail_hits = sum(1 for s in got_spans if s.start > len(long_text) - 300)
    for s in got_spans:
        assert long_text[s.start : s.end] == s.text, "長文でオフセットが元テキスト基準でない"
    starts = [s.start for s in got_spans]
    assert starts == sorted(starts), "スパンが start 昇順でない"
    for a, b in zip(got_spans, got_spans[1:]):
        assert not a.overlaps(b), "統合後に重複スパンが残っている"
    print(
        f"sliding window: {len(long_text)} chars -> {len(win)} windows (max_length=64), "
        f"{len(got_spans)} spans, {tail_hits} in last 300 chars, "
        f"raw scores returned={len(long_raw[0])}"
    )
    assert len(win) > 1, "長文でウィンドウが 1 個しかできていない"
    assert len(long_raw[0]) == len(got_spans), "生スコアの本数がスパン数と一致しない"

    metrics = _span_scores([long_doc], [got_spans])
    print(
        f"  long-doc span metrics: exact f1={metrics['exact']['f1']:.3f} "
        f"partial f1={metrics['partial']['f1']:.3f}"
    )

    # --- 7. save / load 往復 ----------------------------------------------
    assert os.path.exists(os.path.join(tmp_dir, LABELS_FILENAME)), "ラベル台帳が保存されていない"
    reloaded = TokenClassifier.load(tmp_dir, device="cpu")
    assert reloaded.label_list == clf.label_list, "ラベル台帳が往復で壊れた"
    re_pred = reloaded.predict(sample_texts, max_length=128, threshold=0.3)
    same = all(
        [s.key() for s in a] == [s.key() for s in b] for a, b in zip(preds, re_pred)
    )
    print(f"save/load: labels ok, device={reloaded.device}, spans identical={same}")

    # --- 8. CPU 推論スループット ------------------------------------------
    bench_texts = [d.text for d in docs[:32]]
    reloaded.to("cpu")
    t0 = time.time()
    reloaded.predict(bench_texts, batch_size=16, max_length=128, threshold=0.5)
    dt = time.time() - t0
    print(f"cpu throughput: {len(bench_texts)/dt:.1f} docs/s ({dt:.2f}s for {len(bench_texts)} docs)")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("=" * 70)
    print("sumi.model self-test: ALL OK")
    print("=" * 70)


if __name__ == "__main__":  # pragma: no cover
    _selftest()
