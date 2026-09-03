"""Sumi のコマンドラインインターフェース。

Claim: 可逆性 / 検出率 / CPU速度 — 実運用の最小単位である
「文書を墨消しして、対応表を手元に残し、あとで戻す」を1コマンドで行えるようにする。

    sumi redact input.txt --out masked.txt --map map.json
    sumi restore masked.txt --map map.json --out restored.txt
    sumi detect input.txt --json
    sumi eval --data data/dataset --model artifacts/sumi-model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Sequence

from sumi.detector import DEFAULT_MODEL_DIR, SumiDetector
from sumi.mask import ReversibleMasker
from sumi.types import ALL_TYPES, PIIType, Span


def _read(path: str) -> str:
    """入力を読む (``-`` は標準入力)。

    Claim: 可逆性 — パイプ経由でも同じ経路を通ることを保証する。
    """
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path: str | None, content: str) -> None:
    """出力を書く (``None``/``-`` は標準出力)。

    Claim: 可逆性 — 書き出し経路を1点に集約する。
    """
    if not path or path == "-":
        sys.stdout.write(content)
        return
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _build_detector(args) -> SumiDetector:
    """引数から検出器を組み立てる。

    Claim: 検出率 / 低誤検出 — 閾値・層の有無・量子化の選択を利用者に委ね、
    「検出漏れの可能性を前提に、用途に応じて調整できる」設計を守る。
    """
    calibrator = None
    cal_path = os.path.join(args.model or DEFAULT_MODEL_DIR, "calibrator.json")
    if not getattr(args, "no_calibration", False) and os.path.exists(cal_path):
        from sumi.calibrate import SpanCalibrator

        calibrator = SpanCalibrator.load(cal_path)
    return SumiDetector(
        args.model,
        use_rules=not getattr(args, "no_rules", False),
        use_model=not getattr(args, "no_model", False),
        threshold=args.threshold,
        calibrator=calibrator,
        device=getattr(args, "device", "cpu"),
        onnx=getattr(args, "onnx", False),
    )


def cmd_redact(args) -> int:
    """PII を検出して墨消しし、対応表を保存する。

    Claim: 可逆性 — 主要ユースケース。マスク済み本文と対応表を分離して出力し、
    対応表はローカルに 0600 で残す (外部に送るのはマスク済み本文だけ)。
    """
    text = _read(args.input)
    det = _build_detector(args)
    t0 = time.perf_counter()
    masked, mmap = det.redact(text, doc_id=os.path.basename(args.input))
    dt = time.perf_counter() - t0

    _write(args.out, masked)
    masker = ReversibleMasker()
    if args.map:
        masker.save_map(mmap, args.map)

    if not args.quiet:
        n = len(mmap.entries)
        by_type: dict[str, int] = {}
        for e in mmap.entries:
            by_type[e.label.en] = by_type.get(e.label.en, 0) + 1
        summary = ", ".join(f"{k}×{v}" for k, v in sorted(by_type.items()))
        print(f"[sumi] redacted {n} span(s) ({summary or 'none'})  {dt*1000:.0f}ms",
              file=sys.stderr)
        if args.map:
            print(f"[sumi] mapping -> {args.map} (mode 0600, stays on this machine)",
                  file=sys.stderr)
        print("[sumi] Note: Sumi is not a complete detector. Assume misses happen, and "
              "choose the threshold and use case accordingly.", file=sys.stderr)
    return 0


def cmd_restore(args) -> int:
    """対応表を使ってマスク済みテキストを復元する。

    Claim: 可逆性 — LLM から戻ってきたテキストを元の値に戻す経路。
    """
    text = _read(args.input)
    mmap = ReversibleMasker.load_map(args.map)
    restored = ReversibleMasker().unmask(text, mmap)
    _write(args.out, restored)
    if not args.quiet:
        remaining = sum(1 for p in mmap.placeholders() if p in restored)
        print(f"[sumi] restored {len(mmap.entries)} span(s)"
              + (f" ({remaining} placeholder(s) left unresolved)" if remaining else ""),
              file=sys.stderr)
    return 0


def cmd_detect(args) -> int:
    """検出結果を表示する (JSON も可)。

    Claim: 検出率 — 何がどのスコアでどの層から出たかを確認できるようにする。
    """
    text = _read(args.input)
    det = _build_detector(args)
    res = det.detect_result(text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if not res.spans:
        print("(nothing detected)")
    for s in res.spans:
        src = s.meta.get("from", s.source.value)
        rule = f" rule={s.meta.get('rule_id')}" if s.meta.get("rule_id") else ""
        print(f"{s.start:5d}-{s.end:<5d} {s.label.en:13s} {s.score:5.3f} [{src}]{rule}  {s.text!r}")
    t = res.timings
    print(f"\nrules {t['rules']*1000:.1f}ms / model {t['model']*1000:.1f}ms "
          f"/ merge {t['merge']*1000:.1f}ms = {t['total']*1000:.1f}ms total", file=sys.stderr)
    return 0


def cmd_eval(args) -> int:
    """テストセットと否定例サブセットで評価する。

    Claim: 検出率 / 低誤検出 — 手元の構成で主要指標を再計算できるようにする。
    """
    from scripts.build_dataset import read_jsonl
    from sumi.calibrate import (
        detection_rates, false_positive_report, recall_at_fixed_fpr,
    )

    test = read_jsonl(os.path.join(args.data, "test.jsonl"))[: args.limit or None]
    negs = read_jsonl(os.path.join(args.data, "negatives.jsonl"))[: args.limit or None]
    det = _build_detector(args)

    t0 = time.perf_counter()
    pred = det.detect_batch([d.text for d in test])
    npred = det.detect_batch([d.text for d in negs])
    dt = time.perf_counter() - t0

    rates = detection_rates(test, pred, mode=args.match)
    fpr = false_positive_report(negs, npred)
    r_at = recall_at_fixed_fpr(test, pred, negs, npred, target_fpr=args.target_fpr,
                               mode=args.match)

    print(f"{len(test)}+{len(negs)} documents / {dt:.1f}s "
          f"({(len(test)+len(negs))/dt:.1f} docs/s)")
    print("\nRecall by type (partial match)")
    print("-" * 58)
    for t in ALL_TYPES:
        r = rates["by_type"].get(t.value)
        if not r or r["support"] == 0:
            continue
        print(f"  {t.en:13s} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f}  (n={r['support']})")
    print("-" * 58)
    print(f"  micro    P={rates['micro']['precision']:.3f} R={rates['micro']['recall']:.3f} "
          f"F1={rates['micro']['f1']:.3f}")
    print(f"\nFalse positives on the negative subset")
    print(f"  false positives per document   {fpr['fp_per_doc']:.4f}")
    print(f"  document-level FP rate         {fpr['doc_level_fp_rate']:.4f}")
    print(f"  per 1000 characters            {fpr['fp_per_1000_chars']:.4f}")
    print(f"\nRecall at a fixed false-positive budget of {args.target_fpr:.0%}")
    print(f"  threshold {r_at['threshold']:.3f} / actual FPR {r_at['fpr']:.4f} "
          f"/ overall recall {r_at['overall_recall']:.3f}")
    for k, v in sorted(r_at.get("by_type", {}).items()):
        if v.get("support"):
            print(f"    {PIIType(k).en:13s} {v['recall']:.3f}  (n={v['support']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """引数パーサを構築する。

    Claim: 可逆性 — CLI の契約 (``sumi redact input.txt --out masked.txt --map map.json``)
    を定義する。
    """
    p = argparse.ArgumentParser(
        prog="sumi",
        description="Sumi — Japanese PII detection and reversible masking. "
                    "Not a complete detector: assume misses and choose your threshold.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", default=None, help=f"model directory (default {DEFAULT_MODEL_DIR})")
        sp.add_argument("--threshold", type=float, default=0.5, help="minimum calibrated score to keep")
        sp.add_argument("--no-rules", action="store_true", help="disable the rule layer")
        sp.add_argument("--no-model", action="store_true", help="disable the model layer")
        sp.add_argument("--no-calibration", action="store_true", help="disable the calibrator")
        sp.add_argument("--device", default="cpu", choices=["cpu", "mps"])
        sp.add_argument("--onnx", action="store_true", help="run the ONNX model instead of PyTorch")
        sp.add_argument("-q", "--quiet", action="store_true")

    sp = sub.add_parser("redact", help="redact PII and save the mapping table")
    sp.add_argument("input", help="input file (- for stdin)")
    sp.add_argument("--out", default="-", help="where to write the masked text")
    sp.add_argument("--map", default="", help="where to write the mapping (JSON, mode 0600)")
    common(sp)
    sp.set_defaults(func=cmd_redact)

    sp = sub.add_parser("restore", help="restore original values using the mapping")
    sp.add_argument("input", help="masked text (- for stdin)")
    sp.add_argument("--map", required=True, help="mapping JSON")
    sp.add_argument("--out", default="-")
    sp.add_argument("-q", "--quiet", action="store_true")
    sp.set_defaults(func=cmd_restore)

    sp = sub.add_parser("detect", help="show what would be detected")
    sp.add_argument("input")
    sp.add_argument("--json", action="store_true")
    common(sp)
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("eval", help="compute the headline metrics on test + negatives")
    sp.add_argument("--data", default="data/dataset")
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--match", default="partial", choices=["partial", "exact"])
    sp.add_argument("--target-fpr", type=float, default=0.05)
    common(sp)
    sp.set_defaults(func=cmd_eval)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """エントリポイント。

    Claim: 可逆性 — ``sumi`` コマンドの本体。
    """
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
