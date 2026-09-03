"""較正と主要指標 (誤検出率固定時の検出率) を検証する。

Claim: 較正 / 低誤検出 — スパン確率が「真である確率」として意味を持つこと、
および主要指標が定義どおりに動くことを確認する。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sumi.calibrate import (
    SpanCalibrator,
    detection_rates,
    expected_calibration_error,
    false_positive_report,
    match_spans,
    recall_at_fixed_fpr,
)
from sumi.types import Document, PIIType, Span


def _miscalibrated(n=3000, seed=0):
    """真の確率から乖離したスコアを作る (過信モデルの模擬)。

    Claim: 較正 — 較正器が実際に ECE を下げることを確かめるための素材。
    """
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    labels = (rng.uniform(size=n) < true_p).astype(int)
    logit = np.log(true_p / (1 - true_p))
    scores = 1 / (1 + np.exp(-logit * 2.2))     # 温度 <1 相当 = 過信
    return scores.tolist(), labels.tolist()


@pytest.mark.parametrize("method", ["temperature", "isotonic"])
def test_calibration_reduces_ece(method):
    """較正で ECE が下がること。

    Claim: 較正 — 較正が「効いている」ことの最小の定義。
    """
    scores, labels = _miscalibrated()
    before = expected_calibration_error(scores, labels)
    cal = SpanCalibrator(method=method).fit(scores, labels)
    after = expected_calibration_error(cal.transform(scores), labels)
    assert after < before, f"{method}: ECE が下がっていない ({before:.4f} -> {after:.4f})"


def test_calibrator_roundtrip(tmp_path):
    """較正器を保存して読み戻しても同じ変換をすること。

    Claim: 較正 — 学習時に当てはめた較正を推論時に再現できることを保証する。
    """
    scores, labels = _miscalibrated(800, seed=3)
    cal = SpanCalibrator(method="temperature").fit(scores, labels)
    p = tmp_path / "cal.json"
    cal.save(str(p))
    again = SpanCalibrator.load(str(p))
    a = cal.transform(scores[:50])
    b = again.transform(scores[:50])
    assert all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def test_calibrated_scores_stay_probabilities():
    """較正後のスコアが [0,1] に収まること。

    Claim: 較正 — 閾値の意味が壊れないことを保証する。
    """
    scores, labels = _miscalibrated(500, seed=7)
    for m in ("temperature", "isotonic"):
        out = SpanCalibrator(method=m).fit(scores, labels).transform(scores)
        assert all(0.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in out)


# ------------------------------------------------------------ 突き合わせ


def test_match_spans_exact_vs_partial():
    """exact と partial の突き合わせが定義どおりに動くこと。

    Claim: 検出率 — 境界のゆれをどう数えるかが検出率を左右するため。
    """
    gold = [Span(0, 4, PIIType.NAME, "田中太郎")]
    pred_exact = [Span(0, 4, PIIType.NAME, "田中太郎", 0.9)]
    pred_partial = [Span(0, 5, PIIType.NAME, "田中太郎様", 0.9)]

    tp, fp, fn = match_spans(gold, pred_exact, mode="exact")
    assert len(tp) == 1 and not fp and not fn
    tp, fp, fn = match_spans(gold, pred_partial, mode="exact")
    assert not tp and len(fp) == 1 and len(fn) == 1
    tp, fp, fn = match_spans(gold, pred_partial, mode="partial")
    assert len(tp) == 1 and not fp and not fn


def test_one_gold_consumes_one_pred():
    """1つの正解が複数の予測を吸収しないこと。

    Claim: 低誤検出 — 重複予測を検出扱いにすると誤検出が過小評価される。
    """
    gold = [Span(0, 4, PIIType.NAME, "田中太郎")]
    preds = [
        Span(0, 4, PIIType.NAME, "田中太郎", 0.9),
        Span(0, 2, PIIType.NAME, "田中", 0.8),
    ]
    tp, fp, fn = match_spans(gold, preds, mode="partial")
    assert len(tp) == 1
    assert len(fp) == 1, "余分な予測が誤検出に数えられていない"


def test_false_positive_report_definitions():
    """3つの誤検出率の定義が正しく計算されること。

    Claim: 低誤検出 — README が引用する数値の定義を固定する。
    """
    negs = [
        Document(text="あ" * 100, spans=[], doc_id="n0", subset="negatives"),
        Document(text="い" * 100, spans=[], doc_id="n1", subset="negatives"),
        Document(text="う" * 100, spans=[], doc_id="n2", subset="negatives"),
        Document(text="え" * 100, spans=[], doc_id="n3", subset="negatives"),
    ]
    preds = [
        [Span(0, 2, PIIType.NAME, "ああ", 0.9), Span(5, 7, PIIType.NAME, "ああ", 0.8)],
        [],
        [Span(0, 2, PIIType.NAME, "うう", 0.7)],
        [],
    ]
    rep = false_positive_report(negs, preds)
    assert rep["n_fp"] == 3
    assert rep["fp_per_doc"] == pytest.approx(3 / 4)
    assert rep["doc_level_fp_rate"] == pytest.approx(2 / 4)
    assert rep["fp_per_1000_chars"] == pytest.approx(3 / (400 / 1000))


# --------------------------------------------------------- 主要指標


def test_recall_at_fixed_fpr_respects_target():
    """誤検出率を固定したときの検出率が、実際に目標FPR以下で得られること。

    Claim: 低誤検出 — 本プロジェクトの主要指標が定義どおりに動くことの確認。
    """
    pos = [Document(text="田中太郎と佐藤花子", doc_id="p0",
                    spans=[Span(0, 4, PIIType.NAME, "田中太郎"),
                           Span(5, 9, PIIType.NAME, "佐藤花子")])]
    # 高スコアの正解と、低スコアの正解
    pos_pred = [[Span(0, 4, PIIType.NAME, "田中太郎", 0.95),
                 Span(5, 9, PIIType.NAME, "佐藤花子", 0.30)]]
    negs = [Document(text="森の中を歩く", spans=[], doc_id=f"n{i}", subset="negatives")
            for i in range(10)]
    # 否定例には低スコアの誤検出だけがある
    neg_pred = [[Span(0, 1, PIIType.NAME, "森", 0.40)] if i < 5 else [] for i in range(10)]

    r = recall_at_fixed_fpr(pos, pos_pred, negs, neg_pred, target_fpr=0.0, mode="partial")
    assert r["fpr"] <= 1e-9, "目標FPR を満たしていない"
    # 閾値が 0.40 より上に上がるので、0.30 の正解は落ち、0.95 は残る
    assert r["overall_recall"] == pytest.approx(0.5)
    assert r["threshold"] > 0.40


def test_recall_at_fixed_fpr_is_monotone_in_target():
    """許容FPRを緩めると検出率が下がらないこと (単調性)。

    Claim: 低誤検出 — トレードオフ曲線が破綻していないことの確認。
    """
    rng = np.random.default_rng(11)
    pos, pos_pred = [], []
    for i in range(60):
        t = "田中太郎さんと佐藤花子さん"
        pos.append(Document(text=t, doc_id=f"p{i}",
                            spans=[Span(0, 4, PIIType.NAME, "田中太郎")]))
        pos_pred.append([Span(0, 4, PIIType.NAME, "田中太郎", float(rng.uniform(0.3, 1.0)))])
    negs, neg_pred = [], []
    for i in range(60):
        negs.append(Document(text="森の中を歩く", spans=[], doc_id=f"n{i}",
                             subset="negatives"))
        neg_pred.append([Span(0, 1, PIIType.NAME, "森", float(rng.uniform(0.1, 0.8)))]
                        if i % 2 == 0 else [])

    prev = -1.0
    for tgt in (0.0, 0.05, 0.1, 0.2, 0.5):
        r = recall_at_fixed_fpr(pos, pos_pred, negs, neg_pred, target_fpr=tgt,
                                mode="partial")
        assert r["overall_recall"] >= prev - 1e-9, (
            f"target={tgt}: recall が単調でない ({prev} -> {r['overall_recall']})"
        )
        prev = r["overall_recall"]


def test_recall_at_fixed_fpr_degenerate_cases():
    """退化ケースで例外を投げずに旗を立てること。

    Claim: 低誤検出 — 否定例0件・予測0件などで評価が落ちないようにする。
    """
    pos = [Document(text="田中太郎", doc_id="p0",
                    spans=[Span(0, 4, PIIType.NAME, "田中太郎")])]
    r = recall_at_fixed_fpr(pos, [[]], [], [], target_fpr=0.05)
    assert r["overall_recall"] == 0.0
    r2 = recall_at_fixed_fpr(pos, [[Span(0, 4, PIIType.NAME, "田中太郎", 0.9)]],
                             [], [], target_fpr=0.05)
    assert r2["overall_recall"] == pytest.approx(1.0)


def test_detection_rates_by_type():
    """種別ごとの precision/recall/f1 が正しいこと。

    Claim: 検出率 — 図1が引用する数値の計算を固定する。
    """
    docs = [Document(text="田中太郎 090-1234-5678", doc_id="d0",
                     spans=[Span(0, 4, PIIType.NAME, "田中太郎"),
                            Span(5, 18, PIIType.PHONE, "090-1234-5678")])]
    preds = [[Span(0, 4, PIIType.NAME, "田中太郎", 0.9)]]   # 電話は取りこぼし
    r = detection_rates(docs, preds, mode="partial")
    assert r["by_type"]["NAME"]["recall"] == pytest.approx(1.0)
    assert r["by_type"]["PHONE"]["recall"] == pytest.approx(0.0)
    assert r["micro"]["recall"] == pytest.approx(0.5)
    assert r["micro"]["precision"] == pytest.approx(1.0)
