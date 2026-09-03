"""否定例サブセットでの誤検出率が閾値以下であることを検証する。

Claim: 低誤検出 — 主張(4)。本プロジェクトの主戦場である
「紛らわしい否定例での誤検出」を、閾値つきの回帰テストとして固定する。

学習済みモデルが無い環境でも意味のある検査になるよう、
規則層のみの誤検出率は必ず検査し、モデルがある場合は統合構成も検査する。
"""

from __future__ import annotations

import os

import pytest

from sumi.detector import DEFAULT_MODEL_DIR, SumiDetector
from sumi.negatives import (
    NEGATIVE_KINDS,
    HardNegativeGenerator,
    attach_surface_index,
    classify_false_positive,
)
from sumi.rules import RuleLayer
from sumi.types import Document

#: 規則層のみの許容誤検出率 (否定例1文書あたりの誤検出スパン数)
RULES_ONLY_MAX_FP_PER_DOC = 0.02
#: 規則層のみの許容誤検出率 (誤検出が1件以上出た文書の割合)
RULES_ONLY_MAX_DOC_LEVEL = 0.02
#: 統合構成 (モデルあり) の許容文書レベル誤検出率
MERGED_MAX_DOC_LEVEL = 0.25


@pytest.fixture(scope="module")
def negatives() -> list[Document]:
    """否定例文書 (正解スパン0件)。

    Claim: 低誤検出 — ここでの検出は定義上すべて誤検出。
    """
    gen = HardNegativeGenerator(seed=4321)
    docs = gen.build_negative_documents(400)
    attach_surface_index(docs, gen)
    return docs


def _fp_stats(preds, docs) -> dict:
    """誤検出の集計。

    Claim: 低誤検出 — 文書あたり件数と文書レベル率の両方を返す。
    """
    n_fp = sum(len(p) for p in preds)
    n_docs_with = sum(1 for p in preds if p)
    return {
        "n_fp": n_fp,
        "fp_per_doc": n_fp / len(docs),
        "doc_level": n_docs_with / len(docs),
    }


def test_negative_docs_have_no_gold_spans(negatives):
    """否定例サブセットに正解スパンが1件も無いこと。

    Claim: 低誤検出 — この前提が崩れると誤検出率の定義そのものが壊れる。
    """
    assert all(len(d.spans) == 0 for d in negatives)
    assert all(d.subset == "negatives" for d in negatives)


def test_all_negative_kinds_are_represented(negatives):
    """8つの否定例型がすべて生成されていること。

    Claim: 低誤検出 — 特定の型だけで測って良い数字を出すことを防ぐ。
    """
    seen = {k for d in negatives for k in d.negative_kinds}
    missing = set(NEGATIVE_KINDS) - seen
    assert not missing, f"生成されていない否定例型: {sorted(missing)}"


def test_rules_only_false_positive_rate_below_threshold(negatives):
    """規則層のみの誤検出率が閾値以下であること。

    Claim: 低誤検出 — 主張(4)。規則層は「形式が合えば拾う」ため、
    型番・注文番号・議案番号などで暴発しやすい。ここを固定しておく。
    """
    layer = RuleLayer()
    preds = [layer.detect(d.text) for d in negatives]
    st = _fp_stats(preds, negatives)
    if st["n_fp"]:
        kinds: dict[str, int] = {}
        for d, ps in zip(negatives, preds):
            for s in ps:
                k = classify_false_positive(s, d)
                kinds[k] = kinds.get(k, 0) + 1
        detail = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
        examples = [
            (s.label.value, s.text, s.meta.get("rule_id"))
            for d, ps in zip(negatives, preds) for s in ps
        ][:5]
    else:
        detail, examples = "なし", []
    assert st["fp_per_doc"] <= RULES_ONLY_MAX_FP_PER_DOC, (
        f"規則層の誤検出 {st['fp_per_doc']:.4f} 件/文書 > 上限 "
        f"{RULES_ONLY_MAX_FP_PER_DOC}\n  内訳: {detail}\n  例: {examples}"
    )
    assert st["doc_level"] <= RULES_ONLY_MAX_DOC_LEVEL, (
        f"規則層の文書レベル誤検出率 {st['doc_level']:.4f} > 上限 "
        f"{RULES_ONLY_MAX_DOC_LEVEL}\n  内訳: {detail}"
    )


@pytest.mark.skipif(
    not os.path.isdir(DEFAULT_MODEL_DIR),
    reason="学習済みモデルが無い (scripts/train.py 実行後に有効になる)",
)
def test_merged_false_positive_rate_below_threshold(negatives):
    """規則層+モデル層の統合構成の誤検出率が閾値以下であること。

    Claim: 低誤検出 — 主張(4)の本番。氏名・住所はモデルが担当するため、
    普通名詞と同形の姓や地名で暴発しないことをここで固定する。
    """
    det = SumiDetector(DEFAULT_MODEL_DIR, device="cpu")
    preds = det.detect_batch([d.text for d in negatives])
    st = _fp_stats(preds, negatives)
    kinds: dict[str, int] = {}
    for d, ps in zip(negatives, preds):
        for s in ps:
            k = classify_false_positive(s, d)
            kinds[k] = kinds.get(k, 0) + 1
    detail = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
    assert st["doc_level"] <= MERGED_MAX_DOC_LEVEL, (
        f"統合構成の文書レベル誤検出率 {st['doc_level']:.4f} > 上限 "
        f"{MERGED_MAX_DOC_LEVEL}\n  誤検出 {st['n_fp']} 件 / 内訳: {detail}"
    )


def test_injection_preserves_gold_spans():
    """否定例の注入が既存の正解スパンを壊さないこと。

    Claim: 低誤検出 — 座標がずれると学習データが汚染され、
    誤検出率の測定自体が信用できなくなる。
    """
    from sumi.synth import build_documents

    base = build_documents(30, seed=9)
    for d in base:
        before = [(s.label.value, s.text) for s in d.sorted_spans()]
        nd = HardNegativeGenerator(seed=len(d.text) % 97).inject(d, k=3)
        nd.validate()
        assert [(s.label.value, s.text) for s in nd.sorted_spans()] == before
        assert len(nd.text) > len(d.text)


def test_closed_loop_shifts_generation_toward_errors():
    """閉ループが誤検出の多い型へ生成分布を寄せること。

    Claim: 低誤検出 — 「誤りが多い型を狙って次バッチを偏らせる」という
    差別化の中心が実際に機能していることを確認する。
    """
    from collections import Counter

    gen = HardNegativeGenerator(seed=1)
    before = dict(gen.weights)
    fp = {"phone_like_id": 50, "common_noun_surname": 30, "place_as_person": 5}
    after = gen.reweight_from_errors(fp, strength=2.0)

    assert after["phone_like_id"] > before["phone_like_id"]
    assert after["common_noun_surname"] > before["common_noun_surname"]
    assert after["date_like_nondob"] < before["date_like_nondob"]
    assert abs(sum(after.values()) - 1.0) < 1e-9

    lo, hi = 0.25 / len(NEGATIVE_KINDS), 4.0 / len(NEGATIVE_KINDS)
    assert all(lo - 1e-9 <= v <= hi + 1e-9 for v in after.values()), "クリップが効いていない"

    # 生成器は1つだけ作る (ループ内で作り直すと毎回同じ乱数列になる)
    sampler = HardNegativeGenerator(seed=2, weights=after)
    drawn = Counter(sampler.sample().kind for _ in range(3000))
    assert drawn["phone_like_id"] > drawn["date_like_nondob"], "実サンプリングに反映されていない"


def test_classify_false_positive_returns_known_kind(negatives):
    """誤検出の型判定が既知の型を返すこと。

    Claim: 低誤検出 — 閉ループの入力が壊れていないことを確認する。
    """
    import re

    from sumi.types import PIIType, Span

    valid = set(NEGATIVE_KINDS) | {"other"}
    checked = 0
    for d in negatives[:80]:
        m = re.search(r"[一-龥]{2,6}", d.text)
        if not m:
            continue
        k = classify_false_positive(
            Span(m.start(), m.end(), PIIType.NAME, m.group(0)), d
        )
        assert k in valid, f"未知の型を返した: {k}"
        checked += 1
    assert checked >= 40
