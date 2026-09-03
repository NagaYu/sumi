"""Sumi のトークン分類モデルを学習し、較正器を当てはめる。

Claim: 検出率 / 低誤検出 / 較正 — 合成コーパスで PII スパン抽出を学習し、
検証セットでスパン確率を較正する。学習後には否定例サブセットで誤検出を型別に
集計し、``negatives_weights.json`` として **次バッチ生成のための閉ループ入力** を書き出す。

使い方:
    python3 scripts/train.py --epochs 3 --batch-size 16
    python3 scripts/train.py --resume artifacts/sumi-model --epochs 1   # 追加学習
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_dataset import read_jsonl
from sumi.calibrate import SpanCalibrator, expected_calibration_error, match_spans
from sumi.model import TokenClassifier, TrainConfig
from sumi.negatives import classify_false_positive
from sumi.rules import RuleLayer, merge_spans
from sumi.types import RULE_DETERMINISTIC, Document, Span

DATA_DIR = "data/dataset"
OUT_DIR = "artifacts/sumi-model"


def collect_calibration_pairs(
    model: TokenClassifier,
    docs: list[Document],
    *,
    neg_docs: list[Document] | None = None,
    batch_size: int = 32,
    threshold: float = 0.02,
) -> tuple[list[float], list[int]]:
    """検証文書から (スパンスコア, 正解か否か) の組を集める。

    Claim: 較正 — 較正器を当てはめるための教師データ。
    予測スパンが正解と部分一致すれば 1、しなければ 0 とする。

    **否定例文書を必ず混ぜる。** 陽性文書だけで集めると、よく学習された
    モデルでは予測がほぼ全件正解になり、負例クラスが 0 件になって
    較正器が当てはめられない (実際にそうなった)。否定例サブセットでの予測は
    定義上すべて誤り (label=0) なので、較正に必要な負例をここから供給する。

    ``threshold`` は低く取る。較正は「低スコア帯がどれくらい当たらないか」を
    学ぶ作業なので、閾値で切ってから較正すると意味を失う。
    """
    scores: list[float] = []
    labels: list[int] = []

    preds = model.predict([d.text for d in docs], batch_size=batch_size,
                          threshold=threshold, refine=True)
    for d, ps in zip(docs, preds):
        tp_pairs, _fp, _ = match_spans(d.spans, ps, mode="partial")
        matched = {id(p) for _, p in tp_pairs}
        for p in ps:
            scores.append(float(p.score))
            labels.append(1 if id(p) in matched else 0)

    if neg_docs:
        npreds = model.predict([d.text for d in neg_docs], batch_size=batch_size,
                               threshold=threshold, refine=True)
        for ps in npreds:
            for p in ps:
                scores.append(float(p.score))
                labels.append(0)          # 否定例での検出は定義上すべて誤り
    return scores, labels


def false_positive_kinds(
    model: TokenClassifier, neg_docs: list[Document], *,
    threshold: float = 0.5, batch_size: int = 32, use_rules: bool = True,
) -> tuple[Counter, int]:
    """否定例サブセット上の誤検出を型別に数える (閉ループの入力)。

    Claim: 低誤検出 — 「どの紛らわしさに弱いか」を測る。
    ここで数えた誤検出はすべて真の誤検出である (否定例の正解は0件のため)。
    """
    rules = RuleLayer() if use_rules else None
    texts = [d.text for d in neg_docs]
    preds = model.predict(texts, batch_size=batch_size, threshold=threshold, refine=True)
    counts: Counter = Counter()
    total = 0
    for d, ps in zip(neg_docs, preds):
        rs = rules.detect(d.text) if rules else []
        merged = merge_spans(ps, rs, d.text, rule_types=RULE_DETERMINISTIC)
        for s in merged:
            counts[classify_false_positive(s, d)] += 1
            total += 1
    return counts, total


def main() -> None:
    """学習・較正・閉ループ集計を実行する。

    Claim: 検出率 / 低誤検出 / 較正 — 3つを1回の実行で完結させ、
    公開する成果物 (モデル・較正器・次バッチ重み) を揃える。
    """
    ap = argparse.ArgumentParser(description="Sumi モデルを学習する")
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--backbone", default=TrainConfig.backbone)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0,
                    help="検証文書数の上限 (エポックごとの検証コストを抑える)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--calibration", default="temperature",
                    choices=["temperature", "isotonic"])
    ap.add_argument("--calibrate-only", action="store_true",
                    help="学習をやり直さず、保存済みモデルの較正と閉ループ集計だけ実行する")
    args = ap.parse_args()

    print("=" * 74)
    print("Sumi 学習")
    print("=" * 74)

    train = read_jsonl(os.path.join(args.data, "train.jsonl"))
    val = read_jsonl(os.path.join(args.data, "validation.jsonl"))
    negs_path = os.path.join(args.data, "negatives.jsonl")
    negs = read_jsonl(negs_path) if os.path.exists(negs_path) else []
    if args.limit_train:
        train = train[: args.limit_train]
    if args.limit_val:
        val = val[: args.limit_val]
    print(f"train {len(train)} / val {len(val)} / negatives {len(negs)}")
    print(f"正解スパン: train {sum(len(d.spans) for d in train)} / "
          f"val {sum(len(d.spans) for d in val)}")

    cfg = TrainConfig(
        backbone=args.backbone, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, max_length=args.max_length,
        seed=args.seed, device=args.device, output_dir=args.out,
    )
    if args.calibrate_only:
        model = TokenClassifier.load(args.resume or args.out, device="cpu")
    else:
        model = (TokenClassifier.load(args.resume) if args.resume
                 else TokenClassifier.from_backbone(args.backbone))
    print(f"\nbackbone: {args.backbone}")
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"パラメータ数: {n_params/1e6:.1f}M  ({n_params/1e9:.3f}B)")

    if args.calibrate_only:
        print("\n--calibrate-only: 学習をスキップし、保存済みモデルを較正します")
        history, train_secs = {"skipped": True}, 0.0
    else:
        t0 = time.perf_counter()
        history = model.train(train, val, cfg)
        train_secs = time.perf_counter() - t0
        print(f"\n学習時間 {train_secs/60:.1f} 分")

    # ------------------------------------------------------------- 較正
    print("\n" + "-" * 74)
    print("較正 (検証セット上のスパン確率)")
    print("-" * 74)
    scores, labels = collect_calibration_pairs(model, val, neg_docs=negs[:800])
    pos = sum(labels)
    print(f"較正用スパン {len(scores)} 件 (正解一致 {pos} / 不一致 {len(scores)-pos})")
    if len(set(labels)) < 2:
        print("  警告: 片方のクラスしか無いため較正をスキップ")
        cal = None
    else:
        ece_before = expected_calibration_error(scores, labels)
        cal = SpanCalibrator(method=args.calibration).fit(scores, labels)
        ece_after = expected_calibration_error(cal.transform(scores), labels)
        print(f"  ECE {ece_before:.4f} -> {ece_after:.4f} ({args.calibration})")
        cal_path = os.path.join(args.out, "calibrator.json")
        cal.save(cal_path)
        print(f"  -> {cal_path}")

    # --------------------------------------------- 閉ループ: 誤検出の型別集計
    weights_path = ""
    if negs:
        print("\n" + "-" * 74)
        print("閉ループ: 否定例サブセットでの誤検出を型別に集計")
        print("-" * 74)
        counts, total = false_positive_kinds(model, negs)
        print(f"  誤検出 {total} 件 / 否定例 {len(negs)} 文書 "
              f"= {total/len(negs):.4f} 件/文書")
        for k, v in counts.most_common():
            print(f"    {k:24s} {v:5d}")
        from sumi.negatives import HardNegativeGenerator

        gen = HardNegativeGenerator(seed=0)
        new_w = gen.reweight_from_errors(dict(counts), strength=1.5)
        weights_path = os.path.join(args.out, "negatives_weights.json")
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(new_w, f, ensure_ascii=False, indent=2)
        print(f"  次バッチ用の重み -> {weights_path}")
        print("  (build_dataset.py --weights で次ラウンドの生成に反映できる)")

    # ------------------------------------------------------------- 記録
    meta = {
        "backbone": args.backbone,
        "params": n_params,
        "train_docs": len(train),
        "val_docs": len(val),
        "train_seconds": round(train_secs, 1),
        "config": {k: getattr(cfg, k) for k in
                   ("epochs", "lr", "batch_size", "max_length", "seed")},
        "history": history,
        "calibration": args.calibration if cal else None,
        "negatives_weights": weights_path or None,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nモデル -> {args.out}")
    print("=" * 74)


if __name__ == "__main__":
    main()
