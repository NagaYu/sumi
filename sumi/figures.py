"""ベンチマーク結果から README/モデルカード用の図を生成する。

Claim: 検出率 / 低誤検出 / CPU速度 / 較正 — 主張を1枚で確認できる形にする。

目玉の2図:
    figures/fig1_name_detection.png
        氏名の検出で既存構成が 0.5 前後に留まる一方、Sumi が大きく上回ることを示す棒グラフ。
    figures/fig2_size_vs_accuracy.png
        「モデルサイズ vs 精度」の散布図。(C) の4B級と (D)(E) の0.13B級を並べる。

補助図:
    fig3_false_positives.png   紛らわしい否定例での誤検出率 (主戦場)
    fig4_speed.png             CPU スループットとメモリ
    fig5_reliability.png       較正 (reliability diagram)
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sumi.calibrate import use_japanese_font
from sumi.types import ALL_TYPES, PIIType

FIG_DIR = "figures"

#: 条件ごとの色 (Sumi を強調、既存構成は控えめ)
COLORS = {
    "presidio_default": "#9aa5b1",
    "presidio_ginza": "#6b7785",
    "local_llm_4b": "#d08770",
    "sumi_fp32": "#2f6f4e",
    "sumi_int8": "#57a773",
    "sumi_rules_only": "#c0c8d0",
}
SHORT = {
    "presidio_default": "(A) Presidio\n既定",
    "presidio_ginza": "(B) Presidio\n+ GiNZA",
    "local_llm_4b": "(C) ローカルLLM\n4B (Q4)",
    "sumi_fp32": "(D) Sumi\n量子化前",
    "sumi_int8": "(E) Sumi\n量子化後",
    "sumi_rules_only": "(参考) 規則層のみ",
}


def load_report(path: str = "benchmarks/results/benchmark.json") -> dict:
    """ベンチマーク結果 JSON を読み込む。

    Claim: 検出率 — 図が必ず実測値から作られることを保証する
    (図の数値をコードに直書きしない)。
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _setup(figsize=(9, 5)):
    """日本語フォントを設定した Figure/Axes を返す。

    Claim: 検出率 — 日本語ラベルが豆腐にならないようにする
    (フォントが無い環境でも落ちない)。
    """
    use_japanese_font()
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def fig_name_detection(report: dict, out: str = f"{FIG_DIR}/fig1_name_detection.png") -> str:
    """目玉図1: 氏名 (と住所) の検出率を条件間で比較する棒グラフ。

    Claim: 検出率 — 「英語中心の既存ツールが日本語の氏名で落ちる」ことと、
    Sumi がそれを大きく上回ることを、1枚で示す。
    """
    results = report["results"]
    types = [PIIType.NAME, PIIType.ADDRESS]
    fig, ax = _setup((10, 5.4))

    x = np.arange(len(results))
    width = 0.38
    for i, t in enumerate(types):
        vals = []
        for r in results:
            e = r["detection"]["by_type"].get(t.value)
            vals.append(e["recall"] if e and e["support"] else 0.0)
        offs = (i - (len(types) - 1) / 2) * width
        bars = ax.bar(
            x + offs, vals, width,
            color=[COLORS.get(r["condition"], "#888") for r in results],
            alpha=1.0 if i == 0 else 0.55,
            edgecolor="white", linewidth=1.2,
            label=t.ja,
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if i == 0 else "normal")

    ax.axhline(0.5, color="#cc4444", linestyle="--", linewidth=1.1, zorder=0)
    ax.text(len(results) - 0.42, 0.512, "0.5", color="#cc4444", fontsize=9, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(r["condition"], r["condition"]) for r in results], fontsize=9)
    ax.set_ylabel("検出率 (recall)")
    ax.set_ylim(0, 1.08)

    # --- 副題は実測値から生成する (図の文言がデータから乖離しないため) ---
    def _best(cond_pred, t):
        vals = [
            (r["detection"]["by_type"].get(t.value) or {}).get("recall")
            for r in results if cond_pred(r["condition"])
        ]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    is_sumi = lambda c: c.startswith("sumi")
    is_base = lambda c: not c.startswith("sumi")
    bn, bs_ = _best(is_base, PIIType.NAME), _best(is_sumi, PIIType.NAME)
    ba, as_ = _best(is_base, PIIType.ADDRESS), _best(is_sumi, PIIType.ADDRESS)
    parts = []
    if bn is not None and bs_ is not None:
        parts.append(f"氏名 既存最良 {bn:.2f} → Sumi {bs_:.2f}")
    if ba is not None and as_ is not None:
        parts.append(f"住所 既存最良 {ba:.2f} → Sumi {as_:.2f}")
    subtitle = " / ".join(parts) if parts else ""
    ax.set_title(f"日本語の氏名・住所の検出率\n{subtitle}", fontsize=12, pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    ood = "ood" in str(report.get("pos_file", "")) or report.get("tag") == "ood"
    kind = ("テンプレート非依存 (地の文へ挿入・汎化評価)" if ood
            else "テンプレート由来 (学習と同じ文書型)")
    fig.text(0.99, 0.01,
             f"陽性 {report['n_pos']} 文書 / partial 一致 / 合成データ / {kind}",
             ha="right", fontsize=7.5, color="#666666")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_size_vs_accuracy(
    report: dict, out: str = f"{FIG_DIR}/fig2_size_vs_accuracy.png"
) -> str:
    """目玉図2: モデルサイズ vs 精度の散布図。

    Claim: 検出率 / CPU速度 — 4B級 LLM (条件C) と 0.13B級 Sumi (条件D/E) を
    同じ平面に置き、「小さくても上回る」ことを示す。

    (D) と (E) はパラメータ数が同じで点が重なるため、マーカー形状を変え、
    ラベルは衝突を検出して上下に振り分け、引き出し線で points と対応づける。
    データ点そのものは動かさない (ジッタで数値を偽らない)。
    """
    results = report["results"]
    fig, ax = _setup((9.5, 6.0))

    MARKERS = {"sumi_int8": "s", "local_llm_4b": "D", "presidio_ginza": "^",
               "presidio_default": "v", "sumi_fp32": "o", "sumi_rules_only": "P"}

    pts = []
    for r in results:
        p = (r["info"].get("params") or 1e7) / 1e9
        f1 = r["detection"]["micro"]["f1"]
        sp = r["speed"]["docs_per_sec"] or 0.1
        pts.append((p, f1, sp, r))

    # 大きいマーカーから先に描く。同じパラメータ数の (D)/(E) が重なっても、
    # 小さい方が上に来るので両方見える。
    sized = [(90 + 380 * min(1.0, np.log10(sp + 1) / 2.0), p, f1, r)
             for p, f1, sp, r in pts]
    for z, (size, p, f1, r) in enumerate(sorted(sized, key=lambda t: -t[0])):
        ax.scatter(p, f1, s=size, color=COLORS.get(r["condition"], "#888"),
                   marker=MARKERS.get(r["condition"], "o"),
                   edgecolor="white", linewidth=1.8, alpha=0.95, zorder=3 + z)

    # --- ラベルの衝突回避: 近接点は上下に振り分けて引き出し線を引く ---
    placed: list[tuple[float, float]] = []
    for p, f1, sp, r in sorted(pts, key=lambda t: (t[0], t[1])):
        lx, ly = np.log10(p), f1
        close = sum(1 for (ax_, ay_) in placed
                    if abs(ax_ - lx) < 0.12 and abs(ay_ - ly) < 0.12)
        # 近接するたびに上下交互 + 距離を広げる
        direction = 1 if close % 2 == 0 else -1
        dy = (34 + 30 * (close // 2)) * direction
        dx = 0 if close == 0 else (26 if direction > 0 else -26)
        label = SHORT.get(r["condition"], r["condition"]).replace("\n", " ")
        q = r["info"].get("quantization", "none")
        qs = "" if q in ("none", "", None) else f" · {q}"
        ax.annotate(
            f"{label}\n{p:.2f}B{qs} / {sp:.1f} docs/s / F1 {f1:.2f}",
            (p, f1), textcoords="offset points", xytext=(dx, dy),
            ha="center", fontsize=8.5, color="#22303c",
            arrowprops=dict(arrowstyle="-", color="#9aa5b1", linewidth=0.9,
                            shrinkA=0, shrinkB=8) if close else None,
        )
        placed.append((lx, ly))

    ax.set_xscale("log")
    ax.set_xlabel("モデルサイズ (パラメータ数, B) — 対数軸")
    ax.set_ylabel("PII検出 micro F1")
    ax.set_ylim(0, 1.18)
    xs = [p for p, _, _, _ in pts]
    ax.set_xlim(min(xs) * 0.45, max(xs) * 2.6)
    ax.set_title("モデルサイズ vs 精度\n点の大きさ = CPU スループット (大きいほど速い)",
                 fontsize=12, pad=12)
    fig.text(0.99, 0.01, "全条件が同一文書集合・同一CPU・同一スレッド数",
             ha="right", fontsize=7.5, color="#666666")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_false_positives(
    report: dict, out: str = f"{FIG_DIR}/fig3_false_positives.png"
) -> str:
    """補助図: 紛らわしい否定例での誤検出率 (主戦場)。

    Claim: 低誤検出 — 「PIIに見えるがPIIでない」文書で、どれだけ踏み止まれるか。
    """
    results = report["results"]
    fig, ax = _setup((9.5, 5))
    vals = [r["false_positives"]["doc_level_fp_rate"] for r in results]
    bars = ax.bar(
        range(len(results)), vals,
        color=[COLORS.get(r["condition"], "#888") for r in results],
        edgecolor="white", linewidth=1.2, width=0.62,
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([SHORT.get(r["condition"], r["condition"]) for r in results], fontsize=9)
    ax.set_ylabel("文書レベル誤検出率 (低いほど良い)")
    ax.set_ylim(0, max(vals + [0.1]) * 1.25)
    ax.set_title("紛らわしい否定例での誤検出率\n"
                 "普通名詞と同形の姓・地名・企業名・型番・施設名などだけで構成した文書",
                 fontsize=12, pad=12)
    ood = "ood" in str(report.get("pos_file", "")) or report.get("tag") == "ood"
    fig.text(0.99, 0.01,
             f"否定例 {report['n_neg']} 文書 (正解スパン0件) / "
             + ("テンプレート非依存の実行" if ood else "テンプレート由来の実行"),
             ha="right", fontsize=7.5, color="#666666")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_speed(report: dict, out: str = f"{FIG_DIR}/fig4_speed.png") -> str:
    """補助図: CPU スループットとピークメモリ。

    Claim: CPU速度 — (C) との差を桁で示す。
    """
    results = report["results"]
    use_japanese_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    names = [SHORT.get(r["condition"], r["condition"]) for r in results]
    cols = [COLORS.get(r["condition"], "#888") for r in results]

    sp = [r["speed"]["docs_per_sec"] for r in results]
    b1 = ax1.bar(range(len(results)), sp, color=cols, edgecolor="white", width=0.62)
    for b, v in zip(b1, sp):
        ax1.text(b.get_x() + b.get_width() / 2, v * 1.06, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_ylabel("スループット (docs/s, 対数軸)")
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(names, fontsize=8.5)
    ax1.set_title("CPU 処理速度", fontsize=11)

    mem = [r["speed"]["peak_rss_mb"] for r in results]
    b2 = ax2.bar(range(len(results)), mem, color=cols, edgecolor="white", width=0.62)
    for b, v in zip(b2, mem):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("ピークメモリ (MB)")
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(names, fontsize=8.5)
    ax2.set_title("ピークメモリ", fontsize=11)

    fig.suptitle("CPU での処理速度とメモリ", fontsize=12.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_reliability(
    report: dict, out: str = f"{FIG_DIR}/fig5_reliability.png",
    scores: Sequence[float] | None = None, labels: Sequence[int] | None = None,
) -> str | None:
    """補助図: Sumi の reliability diagram。

    Claim: 較正 — 出力スコアが「真である確率」としてどれだけ信用できるかを示す。
    """
    from sumi.calibrate import reliability_diagram

    if scores is None or labels is None:
        return None
    fig = reliability_diagram(scores, labels, bins=12,
                              title="Sumi のスパン確率の較正", out_path=out)
    if fig is not None:
        plt.close(fig)
    return out


def fig_recall_at_fpr(
    report: dict, out: str = f"{FIG_DIR}/fig6_recall_at_fpr.png"
) -> str:
    """補助図: 誤検出率を動かしたときの検出率カーブ。

    Claim: 低誤検出 — 運用者が「許容できる誤検出」を選ぶと検出率が決まる、
    という設計思想をそのまま図にする。
    """
    results = report["results"]
    fig, ax = _setup((9, 5.2))
    for r in results:
        curve = r.get("curve") or []
        if len(curve) < 2:
            continue
        pts = sorted({(round(c[1], 5), c[2]) for c in curve})
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.8,
                color=COLORS.get(r["condition"], "#888"),
                label=SHORT.get(r["condition"], r["condition"]).replace("\n", " "))
    ax.axvline(report["target_fpr"], color="#cc4444", linestyle="--", linewidth=1.1)
    ax.text(report["target_fpr"], 0.02, f" 目標 {report['target_fpr']:.0%}",
            color="#cc4444", fontsize=8.5)
    ax.set_xlabel("否定例での誤検出率 (文書あたり)")
    ax.set_ylabel("検出率 (recall)")
    ax.set_ylim(0, 1.05)
    ax.set_title("誤検出率と検出率のトレードオフ", fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_all(report_path: str = "benchmarks/results/benchmark.json",
              suffix: str = "") -> list[str]:
    """全ての図を生成する。

    Claim: 検出率 / 低誤検出 / CPU速度 — README とモデルカードに貼る図を一括生成する。

    Args:
        suffix: 出力ファイル名に付ける接尾辞 (``"_ood"`` など)。
            同じ図をテンプレート由来/非依存の2条件で並べて出せるようにする。
    """
    report = load_report(report_path)
    def f(name):
        return f"{FIG_DIR}/{name}{suffix}.png"
    outs = [
        fig_name_detection(report, f("fig1_name_detection")),
        fig_size_vs_accuracy(report, f("fig2_size_vs_accuracy")),
        fig_false_positives(report, f("fig3_false_positives")),
        fig_speed(report, f("fig4_speed")),
        fig_recall_at_fpr(report, f("fig6_recall_at_fpr")),
    ]
    return [o for o in outs if o]


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/benchmark.json"
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    if not os.path.exists(path):
        print(f"{path} がありません。先に benchmarks/run_benchmark.py を実行してください。")
        raise SystemExit(1)
    for o in build_all(path, suffix):
        print("figure ->", o, f"({os.path.getsize(o)/1024:.0f} KB)")
