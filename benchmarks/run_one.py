"""1つの比較条件だけを独立プロセスで実行し、予測と資源使用量を書き出す。

Claim: CPU速度 / 検出率 — 条件ごとにプロセスを分けることで、
(1) ピークメモリを条件単位で正確に測れ、
(2) ある条件の import が他条件の速度に影響しない、
という2点を保証する。ベンチマークの公平性のための構造。

使い方 (通常は run_benchmark.py から呼ばれる):
    python3 -m benchmarks.run_one --condition sumi_fp32 --docs docs.json --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: CPU 比較を公平にするため、全条件で同じスレッド数に固定する
N_THREADS = int(os.environ.get("SUMI_BENCH_THREADS", "8"))


def peak_rss_mb() -> float:
    """このプロセスのピーク常駐メモリ (MB)。

    Claim: CPU速度 — 「CPUでの処理速度とメモリ」の後半を測る。
    macOS の ru_maxrss はバイト単位で返る。
    """
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024


def build(condition: str, model_dir: str, threshold: float):
    """条件名から baseline インスタンスを作る。

    Claim: 検出率 — 5条件を同じ生成口から作り、取り違えを防ぐ。
    """
    if condition == "presidio_default":
        from benchmarks.baselines.presidio_base import PresidioBaseline

        return PresidioBaseline()
    if condition == "presidio_ginza":
        from benchmarks.baselines.presidio_ginza import PresidioGinzaBaseline

        return PresidioGinzaBaseline()
    if condition == "local_llm_4b":
        from benchmarks.baselines.local_llm import LocalLLMBaseline

        return LocalLLMBaseline()
    if condition in ("sumi_fp32", "sumi_int8"):
        from benchmarks.baselines.sumi_baseline import SumiBaseline

        return SumiBaseline(model_dir, onnx=(condition == "sumi_int8"), threshold=threshold)
    if condition == "sumi_rules_only":
        from benchmarks.baselines.sumi_baseline import SumiBaseline

        return SumiBaseline(
            model_dir, threshold=threshold, use_model=False,
            name="sumi_rules_only", label="(参考) Sumi 規則層のみ",
        )
    raise ValueError(f"unknown condition: {condition}")


def main() -> None:
    """1条件を実行して結果 JSON を書き出す。

    Claim: CPU速度 / 検出率 — 予測スパン・所要時間・ピークメモリを一括で記録する。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--docs", required=True, help="{'pos': [...], 'neg': [...]} の JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="artifacts/sumi-model")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    try:
        import torch

        torch.set_num_threads(N_THREADS)
    except Exception:
        pass
    os.environ.setdefault("SUMI_ORT_THREADS", str(N_THREADS))
    os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))

    with open(args.docs, encoding="utf-8") as f:
        payload = json.load(f)
    pos_texts: list[str] = payload["pos"]
    neg_texts: list[str] = payload["neg"]

    b = build(args.condition, args.model, args.threshold)
    info = b.info()
    if not b.available():
        json.dump(
            {"condition": args.condition, "available": False, "label": info.label},
            open(args.out, "w", encoding="utf-8"), ensure_ascii=False,
        )
        print(f"[{args.condition}] 利用不可 (依存または学習済みモデルが無い)")
        return

    t0 = time.perf_counter()
    b.warmup()
    load_s = time.perf_counter() - t0
    rss_after_load = peak_rss_mb()

    def run(texts: list[str]) -> tuple[list[list[dict]], float]:
        t = time.perf_counter()
        out = [[s.to_dict() for s in b.detect(x)] for x in texts]
        return out, time.perf_counter() - t

    pos_pred, pos_s = run(pos_texts)
    neg_pred, neg_s = run(neg_texts)

    n = len(pos_texts) + len(neg_texts)
    total_s = pos_s + neg_s
    chars = sum(len(t) for t in pos_texts + neg_texts)
    result = {
        "condition": args.condition,
        "available": True,
        "label": info.label,
        "info": {
            "name": info.name, "label": info.label, "params": info.params,
            "runtime": info.runtime, "quantization": info.quantization,
            "notes": info.notes,
        },
        "n_docs": n,
        "n_chars": chars,
        "load_seconds": round(load_s, 3),
        "seconds": round(total_s, 3),
        "docs_per_sec": round(n / total_s, 4) if total_s else None,
        "chars_per_sec": round(chars / total_s, 1) if total_s else None,
        "ms_per_doc": round(1000 * total_s / n, 2) if n else None,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "rss_after_load_mb": round(rss_after_load, 1),
        "threads": N_THREADS,
        "pos_pred": pos_pred,
        "neg_pred": neg_pred,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"[{args.condition}] {n} docs / {total_s:.1f}s "
          f"({result['docs_per_sec']} docs/s, {result['ms_per_doc']} ms/doc) "
          f"peak {result['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
