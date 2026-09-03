"""Sumi の Gradio Space — 墨消し結果・確信度・対応表を、Presidio 単体と並べて表示する。

Claim: 検出率 / 低誤検出 / 可逆性 — 「日本語で何が違うのか」を、
数値ではなく実際の文書上で見せる。Presidio 単体との差分を色分けし、
Sumi だけが拾えた箇所・Presidio だけが拾った箇所 (多くは誤検出) を可視化する。

    python3 app.py                 # ローカル起動
    (Hugging Face Space では自動的に demo.launch() が呼ばれる)
"""

from __future__ import annotations

import html
import json
import os

import gradio as gr

from sumi.detector import DEFAULT_MODEL_DIR, SumiDetector
from sumi.mask import ReversibleMasker
from sumi.types import ALL_TYPES, PIIType, Span  # noqa: F401

MODEL_DIR = os.environ.get("SUMI_MODEL_DIR", DEFAULT_MODEL_DIR)

EXAMPLES = [
    """拝啓 平素より格別のご高配を賜り厚く御礼申し上げます。
このたびは弊社サービスをご利用いただきありがとうございます。
ご登録内容の確認をお願いいたします。

  お名前     森本一郎 様
  生年月日   昭和47年3月4日
  ご住所     〒160-0023 東京都新宿区西新宿2-8-1 グランドビル501
  ご連絡先   090-1234-5678 / morimoto.ichiro@example.co.jp
  会員番号   MB-2024-004512
  引落口座   0001-234-5678901

なお、型番 TX-2024-0355 の製品は生産終了となりました。
契約日は2023年4月1日、次回のご請求は2024-01-15です。""",
    """森の中を歩いていると、林業の振興について泉が湧くように話が広がった。
長野県の気候は寒暖差が大きく、福島の復興も着実に進んでいる。
本田技研工業が新製品を発表し、大和ハウス工業との提携も報じられた。
お客様各位におかれましては、平素より格別のご高配を賜り厚く御礼申し上げます。
会場は仙台市青葉区役所です。議案第12号について採決を行う。
注文番号 0120-8834-221 でお問い合わせください。""",
    """【議事録】
日時   2024年5月20日 14:00-15:30
場所   本社会議室 101-2024
出席者 青木健太、泉さん、大和田課長、株式会社森商事の林様

■決定事項
・申込者 山口花子(平成7年12月1日生)の口座 普通 1234567 を登録する。
・連絡先は 03-1234-5678、予備で 050-9876-5432 とする。
・マイナンバー 1234 5678 9012 の取扱いは別途規程による。""",
]

COLORS = {
    "both": "#cfe8d5",      # 両方が検出
    "sumi": "#a8d5b5",      # Sumi のみ
    "presidio": "#f3c9b6",  # Presidio のみ
}


def _detector() -> SumiDetector:
    """Sumi 検出器を1回だけ構築して使い回す。

    Claim: CPU速度 — Space 上でリクエストごとにモデルを読み直さない。
    """
    global _DET
    try:
        return _DET
    except NameError:
        pass
    calibrator = None
    cal = os.path.join(MODEL_DIR, "calibrator.json")
    if os.path.exists(cal):
        from sumi.calibrate import SpanCalibrator

        calibrator = SpanCalibrator.load(cal)
    _DET = SumiDetector(MODEL_DIR if os.path.isdir(MODEL_DIR) else None,
                        calibrator=calibrator, device="cpu")
    return _DET


def _presidio_only(text: str) -> list[Span]:
    """Presidio + GiNZA 単体の検出結果 (比較対象)。

    Claim: 検出率 — 「現行の実務」構成との差を同じ画面で見せる。
    藁人形にしないため、GiNZA を足した最良に近い構成を使う。
    """
    try:
        from benchmarks.baselines.presidio_ginza import PresidioGinzaBaseline

        global _PRES
        try:
            b = _PRES
        except NameError:
            b = _PRES = PresidioGinzaBaseline()
        if not b.available():
            return []
        return b.detect(text)
    except Exception:
        return []


def _highlight(text: str, spans: list[Span], other: list[Span]) -> str:
    """スパンを色分けした HTML を作る。

    Claim: 検出率 / 低誤検出 — 「どちらが拾ったか」を色で示し、
    差分をひと目で分かるようにする。
    """
    marks = []
    for s in spans:
        kind = "both" if any(s.overlaps(o) for o in other) else "sumi"
        marks.append((s.start, s.end, s, kind))
    for o in other:
        if not any(o.overlaps(s) for s in spans):
            marks.append((o.start, o.end, o, "presidio"))
    marks.sort(key=lambda m: (m[0], m[1]))

    out = []
    pos = 0
    for a, b, s, kind in marks:
        if a < pos:
            continue
        out.append(html.escape(text[pos:a]).replace("\n", "<br>"))
        tip = f"{s.label.en} / score={s.score:.2f} / {kind}"
        out.append(
            f'<span style="background:{COLORS[kind]};border-radius:3px;padding:1px 3px;'
            f'border-bottom:2px solid rgba(0,0,0,.18)" title="{html.escape(tip)}">'
            f"{html.escape(text[a:b])}"
            f'<sub style="font-size:9px;opacity:.65"> {s.label.en}</sub></span>'
        )
        pos = b
    out.append(html.escape(text[pos:]).replace("\n", "<br>"))
    legend = (
        '<div style="margin-bottom:10px;font-size:12px">'
        f'<span style="background:{COLORS["both"]};padding:2px 6px;border-radius:3px">both agree</span> '
        f'<span style="background:{COLORS["sumi"]};padding:2px 6px;border-radius:3px">Sumi only</span> '
        f'<span style="background:{COLORS["presidio"]};padding:2px 6px;border-radius:3px">Presidio+GiNZA only</span>'
        "</div>"
    )
    return (legend + '<div style="line-height:2;font-family:system-ui,sans-serif;'
            'font-size:14px;white-space:normal">' + "".join(out) + "</div>")


def analyze(text: str, threshold: float, use_rules: bool, use_model: bool):
    """入力文書を解析し、墨消し結果・確信度・対応表・比較を返す。

    Claim: 検出率 / 低誤検出 / 可逆性 — 1回の操作で
    (1) 何が隠れたか (2) どれくらい自信があるか (3) どう戻せるか
    (4) 既存構成と何が違うか、をすべて見せる。
    """
    if not text or not text.strip():
        return "", "", [], [], ""

    det = _detector()
    det.threshold = threshold
    # 較正前に切らないよう、モデル層の下限も閾値以下に保つ
    det.model_threshold = min(0.05, threshold)
    if det.rules is not None and not use_rules:
        rules_backup, det.rules = det.rules, None
    else:
        rules_backup = None
    saved_use_model = det.use_model
    if not use_model:
        det.use_model = False

    try:
        res = det.detect_result(text)
        spans = res.spans
    finally:
        if rules_backup is not None:
            det.rules = rules_backup
        det.use_model = saved_use_model

    masked, mmap = ReversibleMasker().mask(res.text, spans)
    pres = _presidio_only(res.text)

    conf_rows = [
        [s.label.en, s.text, round(s.score, 3),
         s.meta.get("from", s.source.value), f"{s.start}-{s.end}"]
        for s in spans
    ]
    map_rows = [
        [r["placeholder"], PIIType(r["label"]).en, r["preview"], r["length"],
         f"{r['start']}-{r['end']}"]
        for r in mmap.redact_summary()
    ]

    only_sumi = [s for s in spans if not any(s.overlaps(o) for o in pres)]
    only_pres = [o for o in pres if not any(o.overlaps(s) for s in spans)]
    t = res.timings
    summary = (
        f"**{len(spans)} span(s) detected**  "
        f"({', '.join(sorted({s.label.en for s in spans})) or 'none'})\n\n"
        f"- Found by **Sumi only**: **{len(only_sumi)}** "
        f"({', '.join(f'{s.label.en} “{s.text}”' for s in only_sumi[:6]) or 'none'})\n"
        f"- Found by **Presidio+GiNZA only**: **{len(only_pres)}** "
        f"({', '.join(f'{o.label.en} “{o.text}”' for o in only_pres[:6]) or 'none'})\n"
        f"- Time: rules {t['rules']*1000:.1f}ms / model {t['model']*1000:.1f}ms "
        f"/ merge {t['merge']*1000:.1f}ms = **{t['total']*1000:.1f}ms total** (CPU)\n\n"
        f"> Sumi is not a complete detector. Assume misses happen, and choose the "
        f"threshold and use case accordingly."
    )
    return masked, _highlight(res.text, spans, pres), conf_rows, map_rows, summary


def build_demo() -> gr.Blocks:
    """Gradio UI を組み立てる。

    Claim: 可逆性 — 対応表は「元値を伏せた要約」として表示し、
    画面経由で元値が漏れない設計をそのまま UI に反映する。
    """
    with gr.Blocks(title="Sumi — Japanese PII detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 墨 Sumi — Japanese PII detection\n"
            "A 0.13B token classifier plus a rule layer, running on CPU. It finds "
            "Japanese names, addresses, phone numbers, dates of birth and ID/number "
            "families, and replaces them with **reversible placeholders**. "
            "The mapping table stays on your machine.\n\n"
            "⚠️ **This does not guarantee legal or regulatory compliance.** It reduces "
            "risk; it is not a complete detector. Assume misses happen."
        )
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Input document (Japanese)", lines=16, value=EXAMPLES[0])
                with gr.Row():
                    thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Threshold")
                    rules_cb = gr.Checkbox(value=True, label="Rule layer")
                    model_cb = gr.Checkbox(value=True, label="Model layer")
                btn = gr.Button("Analyse", variant="primary")
                gr.Examples(
                    examples=[[e] for e in EXAMPLES], inputs=[inp],
                    label="Examples (the second contains no PII at all — only look-alikes)",
                )
            with gr.Column(scale=4):
                summary = gr.Markdown()
                diff = gr.HTML(label="Difference against Presidio + GiNZA")
        with gr.Row():
            masked_out = gr.Textbox(label="Redacted output (this is what you send to an LLM)", lines=12)
            with gr.Column():
                conf = gr.Dataframe(
                    headers=["Type", "Matched text", "Confidence", "Layer", "Offsets"],
                    label="Detections and confidence", wrap=True,
                )
                mp = gr.Dataframe(
                    headers=["Placeholder", "Type", "Masked preview", "Length", "Offsets"],
                    label="Mapping table (original values are never shown here — "
                          "they are written only to a local map.json with mode 0600)",
                    wrap=True,
                )
        gr.Markdown(
            "### Drop into Presidio\n"
            "```python\n"
            "from presidio_analyzer import RecognizerRegistry\n"
            "from sumi.presidio_plugin import register\n"
            "register(registry)   # that is the whole integration\n"
            "```\n"
            "[Code](https://github.com/NagaYu/sumi) · "
            "[Model](https://huggingface.co/NagaYu/sumi-ja-pii) · "
            "[Dataset](https://huggingface.co/datasets/NagaYu/sumi-ja-pii-corpus)"
        )
        btn.click(analyze, [inp, thr, rules_cb, model_cb],
                  [masked_out, diff, conf, mp, summary])
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
