"""5条件 (A)-(E) を同一の文書集合で走らせ、5つの評価軸で比較する。

Claim: 検出率 / 低誤検出 / CPU速度 / 較正 — 本プロジェクトの主張を検証する本体。

評価軸:
    (1) 種別ごとの検出率 (氏名・住所・電話・生年月日・番号類)
    (2) 紛らわしい否定例での誤検出率  ← 主戦場
    (3) 誤検出率を固定したときの検出率
    (4) CPU での処理速度とメモリ
    (5) 較正の良さ

公平性のための設計:
    - 全条件が **まったく同じ文書集合** を処理する (LLM が遅いので集合は共通に絞る)。
    - 条件ごとに独立プロセスで実行し、スレッド数を揃え、ピークメモリを個別に測る。
    - (B) は GiNZA の拡張固有表現を丁寧に写像し住所断片も連結した、
      藁人形ではない最良に近い構成。
    - (C) の JSON パースは壊れた出力からも拾えるようにしてある (LLM に有利側)。

使い方:
    python3 benchmarks/run_benchmark.py --n-pos 300 --n-neg 300
    python3 benchmarks/run_benchmark.py --conditions sumi_fp32,sumi_int8   # 一部だけ
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_dataset import read_jsonl
from sumi.calibrate import (
    detection_rates,
    expected_calibration_error,
    false_positive_report,
    match_spans,
    recall_at_fixed_fpr,
)
from sumi.negatives import classify_false_positive
from sumi.types import ALL_TYPES, PIIType, Span

CONDITIONS = [
    "presidio_default",
    "presidio_ginza",
    "local_llm_4b",
    "sumi_fp32",
    "sumi_int8",
]
OUT_DIR = "benchmarks/results"


def _spans(rows: list[dict]) -> list[Span]:
    """辞書列を Span 列に戻す。

    Claim: 検出率 — 子プロセスの出力を採点可能な形に復元する。
    """
    return [Span.from_dict(r) for r in rows]


def run_condition(cond: str, docs_path: str, model: str, threshold: float,
                  out_dir: str, force: bool, suffix: str = "") -> dict | None:
    """1条件を独立プロセスで実行して結果を読み込む。

    Claim: CPU速度 — プロセス分離により、条件ごとのピークメモリを正しく測る。
    """
    out = os.path.join(out_dir, f"raw_{cond}{suffix}.json")
    if os.path.exists(out) and not force:
        print(f"[{cond}] 既存の結果を再利用: {out}")
    else:
        cmd = [
            sys.executable, "-u", "-m", "benchmarks.run_one",
            "--condition", cond, "--docs", docs_path, "--out", out,
            "--model", model, "--threshold", str(threshold),
        ]
        print(f"[{cond}] 実行中 ...", flush=True)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"[{cond}] 失敗 (rc={p.returncode})")
            print("  " + "\n  ".join((p.stderr or "").strip().splitlines()[-8:]))
            return None
        for line in (p.stdout or "").strip().splitlines():
            if line.startswith("["):
                print("  " + line)
        print(f"  ({time.perf_counter()-t0:.0f}s)")
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def score(raw: dict, pos_docs, neg_docs, *, target_fpr: float, mode: str) -> dict:
    """1条件の生予測を、5つの評価軸に沿って採点する。

    Claim: 検出率 / 低誤検出 / 較正 — 全条件に同一の採点関数を適用する。
    """
    pos_pred = [_spans(r) for r in raw["pos_pred"]]
    neg_pred = [_spans(r) for r in raw["neg_pred"]]

    rates = detection_rates(pos_docs, pos_pred, mode=mode)
    fp = false_positive_report(neg_docs, neg_pred)
    at = recall_at_fixed_fpr(pos_docs, pos_pred, neg_docs, neg_pred,
                             target_fpr=target_fpr, mode=mode)

    # --- 誤検出の型別内訳 (どの紛らわしさで転ぶか) ---
    kinds: dict[str, int] = {}
    for d, ps in zip(neg_docs, neg_pred):
        for s in ps:
            k = classify_false_positive(s, d)
            kinds[k] = kinds.get(k, 0) + 1

    # --- 較正 (スコアを出す条件のみ) ---
    cal = None
    scores: list[float] = []
    labels: list[int] = []
    for d, ps in zip(pos_docs, pos_pred):
        tp_pairs, _, _ = match_spans(d.spans, ps, mode=mode)
        matched = {id(p) for _, p in tp_pairs}
        for p in ps:
            scores.append(float(p.score))
            labels.append(1 if id(p) in matched else 0)
    for d, ps in zip(neg_docs, neg_pred):
        for p in ps:
            scores.append(float(p.score))
            labels.append(0)
    if scores and len(set(labels)) > 1 and len(set(round(s, 4) for s in scores)) > 2:
        cal = {
            "ece": expected_calibration_error(scores, labels),
            "n": len(scores),
            "mean_score": sum(scores) / len(scores),
            "positive_rate": sum(labels) / len(labels),
        }

    return {
        "condition": raw["condition"],
        "label": raw["label"],
        "info": raw["info"],
        "speed": {
            "docs_per_sec": raw["docs_per_sec"],
            "ms_per_doc": raw["ms_per_doc"],
            "chars_per_sec": raw["chars_per_sec"],
            "seconds": raw["seconds"],
            "load_seconds": raw["load_seconds"],
            "peak_rss_mb": raw["peak_rss_mb"],
            "threads": raw["threads"],
        },
        "detection": rates,
        "false_positives": fp,
        "fp_by_kind": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "recall_at_fixed_fpr": {k: v for k, v in at.items() if k != "curve"},
        "curve": at.get("curve", [])[:200],
        "calibration": cal,
    }


def print_report(results: list[dict], target_fpr: float) -> None:
    """人が読む形の比較表を標準出力に出す。

    Claim: 検出率 / 低誤検出 / CPU速度 — 5軸を1画面で見比べられるようにする。
    """
    def cell(v, fmt="{:.3f}", na="  -  "):
        return na if v is None else fmt.format(v)

    W = 30
    print("\n" + "=" * 108)
    print("(1) 種別ごとの検出率 (recall, partial 一致)")
    print("=" * 108)
    # 表示する種別。**ヘッダーと値は必ずこの1つのリストから生成する**
    # (以前、ヘッダーを別の順序で組み立てていたため列がずれていた)
    show = [PIIType.NAME, PIIType.ADDRESS, PIIType.PHONE, PIIType.DOB,
            PIIType.EMAIL, PIIType.BANK_ACCOUNT, PIIType.MYNUMBER, PIIType.MEMBER_ID]
    print(f"{'条件':<{W}}" + "".join(f"{t.ja:>10s}" for t in show))
    for r in results:
        row = f"{r['label']:<{W}}"
        for t in show:
            e = r["detection"]["by_type"].get(t.value)
            row += f"{cell(e['recall'] if e and e['support'] else None):>10s}"
        print(row)

    print("\n" + "=" * 108)
    print("(2) 紛らわしい否定例での誤検出率  ← 主戦場   (低いほど良い)")
    print("=" * 108)
    print(f"{'条件':<{W}}{'文書レベル誤検出率':>20s}{'誤検出/文書':>14s}{'誤検出/1000字':>16s}{'誤検出総数':>12s}")
    for r in results:
        f = r["false_positives"]
        print(f"{r['label']:<{W}}{f['doc_level_fp_rate']:>20.4f}{f['fp_per_doc']:>14.4f}"
              f"{f['fp_per_1000_chars']:>16.3f}{f['n_fp']:>12d}")

    print("\n  誤検出の型別内訳 (どの紛らわしさで転ぶか)")
    for r in results:
        top = list(r["fp_by_kind"].items())[:5]
        s = ", ".join(f"{k}={v}" for k, v in top) or "なし"
        print(f"    {r['label']:<{W}} {s}")

    print("\n" + "=" * 108)
    print(f"(3) 誤検出率を {target_fpr:.0%} に固定したときの検出率")
    print("=" * 108)
    print(f"{'条件':<{W}}{'閾値':>10s}{'実効FPR':>10s}{'全体recall':>12s}{'氏名':>10s}{'住所':>10s}{'達成':>8s}")
    for r in results:
        a = r["recall_at_fixed_fpr"]
        bt = a.get("by_type", {})
        nm = bt.get("NAME", {}).get("recall")
        ad = bt.get("ADDRESS", {}).get("recall")
        print(f"{r['label']:<{W}}{a['threshold']:>10.3f}{a['fpr']:>10.4f}"
              f"{a['overall_recall']:>12.3f}{cell(nm):>10s}{cell(ad):>10s}"
              f"{('○' if a.get('achieved') else '×'):>8s}")

    print("\n" + "=" * 108)
    print("(4) CPU での処理速度とメモリ")
    print("=" * 108)
    print(f"{'条件':<{W}}{'docs/s':>10s}{'ms/doc':>10s}{'ピークRSS(MB)':>16s}{'ロード(s)':>12s}{'規模':>10s}")
    base = next((r for r in results if r["condition"] == "local_llm_4b"), None)
    for r in results:
        sp = r["speed"]
        p = r["info"].get("params")
        psz = f"{p/1e9:.2f}B" if p else "-"
        print(f"{r['label']:<{W}}{sp['docs_per_sec']:>10.2f}{sp['ms_per_doc']:>10.1f}"
              f"{sp['peak_rss_mb']:>16.0f}{sp['load_seconds']:>12.1f}{psz:>10s}")
    if base and base["speed"]["docs_per_sec"]:
        print(f"\n  (C) ローカルLLM 4B に対する速度比:")
        for r in results:
            if r["condition"] == "local_llm_4b":
                continue
            ratio = r["speed"]["docs_per_sec"] / base["speed"]["docs_per_sec"]
            print(f"    {r['label']:<{W}} {ratio:>8.1f} 倍")

    print("\n" + "=" * 108)
    print("(5) 較正の良さ (ECE, 低いほど良い)")
    print("=" * 108)
    for r in results:
        c = r["calibration"]
        if c:
            print(f"  {r['label']:<{W}} ECE={c['ece']:.4f}  "
                  f"(n={c['n']}, 平均score={c['mean_score']:.3f}, 真の割合={c['positive_rate']:.3f})")
        else:
            print(f"  {r['label']:<{W}} スコアが定数のため較正評価は非適用")
    print("=" * 108)


def main() -> None:
    """ベンチマーク全体を実行する。

    Claim: 検出率 / 低誤検出 / CPU速度 / 較正 — 主張の検証を1コマンドで再現可能にする。
    """
    ap = argparse.ArgumentParser(description="Sumi ベンチマーク (5条件 × 5軸)")
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--pos-file", default="test.jsonl",
                    help="陽性側のファイル名。ood.jsonl を指定すると "
                         "テンプレート非依存の汎化評価になる")
    ap.add_argument("--tag", default="", help="結果ファイル名の接尾辞 (例: ood)")
    ap.add_argument("--model", default="artifacts/sumi-model")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--n-pos", type=int, default=300)
    ap.add_argument("--n-neg", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--match", default="partial", choices=["partial", "exact"])
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--force", action="store_true", help="既存結果を無視して再実行")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pos = read_jsonl(os.path.join(args.data, args.pos_file))[: args.n_pos]
    neg = read_jsonl(os.path.join(args.data, "negatives.jsonl"))[: args.n_neg]
    print("=" * 108)
    kind = ("テンプレート非依存 (OOD)" if "ood" in args.pos_file
            else "テンプレート由来 (in-distribution)")
    print(f"Sumi ベンチマーク [{kind}] — 陽性 {len(pos)} 文書 / 否定例 {len(neg)} 文書 "
          f"(全条件が同一集合を処理)")
    print(f"正解スパン {sum(len(d.spans) for d in pos)} 件 / 否定例の正解 0 件")
    print("=" * 108)

    suffix = f"_{args.tag}" if args.tag else ""
    docs_path = os.path.join(args.out, f"_docs{suffix}.json")
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump({"pos": [d.text for d in pos], "neg": [d.text for d in neg]}, f,
                  ensure_ascii=False)

    results = []
    for cond in args.conditions.split(","):
        cond = cond.strip()
        if not cond:
            continue
        raw = run_condition(cond, docs_path, args.model, args.threshold, args.out,
                            args.force, suffix=suffix)
        if raw is None or not raw.get("available"):
            print(f"[{cond}] スキップ (利用不可)")
            continue
        results.append(score(raw, pos, neg, target_fpr=args.target_fpr, mode=args.match))

    if not results:
        print("実行できた条件がありません。")
        return

    print_report(results, args.target_fpr)

    report = {
        "n_pos": len(pos), "n_neg": len(neg),
        "pos_file": args.pos_file, "tag": args.tag,
        "target_fpr": args.target_fpr, "match_mode": args.match,
        "threshold": args.threshold,
        "results": results,
    }
    rp = os.path.join(args.out, f"benchmark{suffix}.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n結果 -> {rp}")


if __name__ == "__main__":
    main()
