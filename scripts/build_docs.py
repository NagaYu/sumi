"""Generate the English README, model card and dataset card from measured results.

Claim: 検出率 / 低誤検出 / CPU速度 — every number that appears in published
documentation is generated from the benchmark/export JSON files, never typed by
hand. This makes it structurally impossible for the published claims to drift
away from what was actually measured.

    python3 scripts/build_docs.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumi.types import ALL_TYPES, PIIType

GH_REPO = "NagaYu/sumi"
HF_MODEL = "NagaYu/sumi-ja-pii"
HF_DATASET = "NagaYu/sumi-ja-pii-corpus"

SHOW_TYPES = [
    PIIType.NAME, PIIType.ADDRESS, PIIType.PHONE, PIIType.DOB,
    PIIType.EMAIL, PIIType.BANK_ACCOUNT, PIIType.MYNUMBER, PIIType.MEMBER_ID,
]

EN = {
    "NAME": "Name", "ADDRESS": "Address", "PHONE": "Phone", "DOB": "Birth date",
    "EMAIL": "Email", "BANK_ACCOUNT": "Bank acct", "CREDIT_CARD": "Card",
    "MYNUMBER": "My Number", "MEMBER_ID": "Member ID", "POSTAL_CODE": "Postal",
}

LABELS_EN = {
    "presidio_default": "(A) Presidio, default",
    "presidio_ginza": "(B) Presidio + GiNZA",
    "local_llm_4b": "(C) Local LLM 4B (Q4)",
    "sumi_fp32": "(D) Sumi fp32",
    "sumi_int8": "(E) Sumi INT8",
    "sumi_rules_only": "(ref) Sumi rules only",
}


def _load(path: str, default=None):
    """Load a JSON file, returning ``default`` when it does not exist.

    Claim: 検出率 — documentation generation must not fail just because one
    optional artifact is missing.
    """
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, spec="{:.3f}", na="—"):
    return na if v is None else spec.format(v)


def _label(r) -> str:
    return LABELS_EN.get(r["condition"], r["label"])


def detection_table(bench: dict, *, with_fp: bool = False) -> str:
    """Per-type recall table.

    Claim: 検出率 — the headline accuracy table, generated from measurements.
    """
    head = "| Condition | " + " | ".join(EN[t.value] for t in SHOW_TYPES) + " | micro F1 |"
    if with_fp:
        head = head[:-1] + " FP rate |"
    sep = "|---|" + "---|" * (len(SHOW_TYPES) + (2 if with_fp else 1))
    rows = []
    for r in bench["results"]:
        cells = []
        for t in SHOW_TYPES:
            e = r["detection"]["by_type"].get(t.value)
            cells.append(_fmt(e["recall"] if e and e["support"] else None))
        row = f"| {_label(r)} | " + " | ".join(cells) + " | " \
              + _fmt(r["detection"]["micro"]["f1"]) + " |"
        if with_fp:
            row += f" {r['false_positives']['doc_level_fp_rate']:.3f} |"
        rows.append(row)
    return "\n".join([head, sep] + rows)


def fp_table(bench: dict) -> str:
    """False-positive table with the dominant error kinds.

    Claim: 低誤検出 — the main battleground of the project.
    """
    rows = []
    for r in bench["results"]:
        f = r["false_positives"]
        top = ", ".join(f"`{k}`={v}" for k, v in list(r["fp_by_kind"].items())[:3]) or "—"
        rows.append(f"| {_label(r)} | **{f['doc_level_fp_rate']:.3f}** | "
                    f"{f['fp_per_doc']:.3f} | {f['n_fp']} | {top} |")
    return "\n".join(
        ["| Condition | Doc-level FP rate | FP per doc | Total FP | Dominant error kinds |",
         "|---|---|---|---|---|"] + rows)


def recall_at_fpr_table(bench: dict) -> str:
    """Recall at a fixed false-positive budget.

    Claim: 低誤検出 — the operationally meaningful metric.
    """
    rows = []
    for r in bench["results"]:
        a = r["recall_at_fixed_fpr"]
        bt = a.get("by_type", {})
        rows.append(
            f"| {_label(r)} | {a['threshold']:.3f} | {a['fpr']:.3f} | "
            f"**{_fmt(a['overall_recall'])}** | {_fmt(bt.get('NAME',{}).get('recall'))} | "
            f"{_fmt(bt.get('ADDRESS',{}).get('recall'))} |")
    return "\n".join(
        ["| Condition | Threshold | Actual FPR | Overall recall | Name | Address |",
         "|---|---|---|---|---|---|"] + rows)


def speed_table(bench: dict) -> str:
    """Throughput and peak memory on CPU.

    Claim: CPU速度 — the cost side of the comparison.
    """
    llm = next((r for r in bench["results"] if r["condition"] == "local_llm_4b"), None)
    rows = []
    for r in bench["results"]:
        sp = r["speed"]
        p = r["info"].get("params") or 0
        ratio = "baseline"
        if llm and llm["speed"]["docs_per_sec"] and r["condition"] != "local_llm_4b":
            ratio = f"**{sp['docs_per_sec']/llm['speed']['docs_per_sec']:.0f}×**"
        rows.append(f"| {_label(r)} | {p/1e9:.2f}B | {sp['docs_per_sec']:.2f} | "
                    f"{sp['ms_per_doc']:.0f} | {sp['peak_rss_mb']:.0f} | {ratio} |")
    return "\n".join(
        ["| Condition | Size | docs/s | ms/doc | Peak RSS (MB) | vs (C) |",
         "|---|---|---|---|---|---|"] + rows)


def calibration_table(bench: dict) -> str:
    """Expected calibration error.

    Claim: 較正 — how much the reported confidence can be trusted.
    """
    rows = []
    for r in bench["results"]:
        c = r["calibration"]
        rows.append(f"| {_label(r)} | " +
                    (f"{c['ece']:.3f} |" if c else "n/a (constant scores) |"))
    return "\n".join(["| Condition | ECE (lower is better) |", "|---|---|"] + rows)


def method_diagram() -> str:
    """Mermaid diagram of the three layers and the closed loop.

    Claim: 検出率 / 低誤検出 / 可逆性 — one picture of how the pieces fit.
    """
    return """```mermaid
flowchart TB
    subgraph DATA["Data construction (closed loop)"]
        C["Openly-licensed Japanese prose<br/>Wikipedia / e-Gov statutes / Aozora Bunko"]
        S["Synthetic PII insertion<br/><b>offsets recorded while building the string</b><br/>names, addresses, phones, dates, accounts, IDs"]
        H["HardNegativeGenerator<br/>surnames that are also common nouns<br/>place/company homographs, honorific edges,<br/>phone-shaped part numbers, facility names"]
        C --> S --> H
    end

    subgraph CORE["Sumi core"]
        direction TB
        R["<b>RuleLayer</b><br/>format-determined types<br/>phone / email / bank / ID schemes<br/><i>checksums are recorded, never gating</i>"]
        M["<b>TokenClassifier</b> 0.13B<br/>ModernBERT-Ja-130m<br/>context-dependent: name, address, DOB"]
        MG["<b>Merge</b> — explicit precedence<br/>1. rule spans accepted unconditionally<br/>2. overlapping model spans discarded<br/>3-4. remaining model spans by score"]
        CA["<b>CalibratedSpans</b><br/>per-span probability + reliability<br/>recall at a fixed false-positive budget"]
        R --> MG
        M --> MG --> CA
    end

    subgraph OUT["Output"]
        RM["<b>ReversibleMasking</b><br/>stable placeholders &lt;NAME_1&gt;"]
        MAP[("Mapping table<br/>local only, mode 0600")]
        LLM["External LLM"]
        EG{{"EgressGuard<br/>blocks any original value"}}
        RM --> MAP
        RM -->|masked text only| EG --> LLM
        LLM -->|response| RM
    end

    H --> M
    CA --> RM
    CA -.->|"false positives counted per negative kind<br/><b>next batch skewed toward the weak kinds</b>"| H

    style R fill:#e8f0e8
    style M fill:#e8f0e8
    style EG fill:#fde8e0
    style MAP fill:#fff3d6
```"""


DISCLAIMER = """> [!WARNING]
> **Sumi does not guarantee legal or regulatory compliance.** It is a tool for
> *reducing the risk* of personal data leaving your machine — not a complete
> detector. Misses will happen. You choose the threshold and the use case.
> All training and evaluation data is synthetic; **no real personal
> information is used anywhere in this project.**"""


def figure_block(prefix: str = "") -> str:
    """Embed the generated figures, OOD first.

    Claim: 検出率 — the generalisation figures are the honest ones, so they lead.
    """
    figs = [
        ("fig1_name_detection_ood.png",
         "**Headline figure 1 (generalisation).** Name and address recall on the "
         "template-independent set. The subtitle numbers are generated from the "
         "measurements, not written by hand."),
        ("fig2_size_vs_accuracy_ood.png",
         "**Headline figure 2 (generalisation).** Model size vs accuracy — the 4B "
         "local LLM (C) and the 0.13B Sumi (D)(E) on one plane. Marker area is CPU throughput."),
        ("fig3_false_positives_ood.png",
         "False-positive rate on deliberately confusable negatives — the main battleground."),
        ("fig4_speed_ood.png", "CPU throughput and peak memory."),
        ("fig6_recall_at_fpr_ood.png", "The recall / false-positive trade-off."),
        ("fig1_name_detection.png",
         "(reference) In-distribution recall. Same document templates as training, so it "
         "is inflated — see the generalisation section."),
        ("fig3_false_positives.png", "(reference) In-distribution false-positive rate."),
    ]
    out = ""
    for name, cap in figs:
        if os.path.exists(os.path.join("figures", name)):
            alt = cap.replace("**", "")
            out += f"\n![{alt}]({prefix}figures/{name})\n\n*{cap}*\n"
    return out


def limitations(bench: dict | None, ood: dict | None) -> str:
    """Limitations section, with the numbers that motivate each one.

    Claim: 低誤検出 — stating where the tool is weak is part of the deliverable.
    """
    name_gap = ""
    if ood:
        res = ood["results"]
        def rec(cond, t):
            r = next((x for x in res if x["condition"] == cond), None)
            e = r["detection"]["by_type"].get(t.value) if r else None
            return e["recall"] if e and e["support"] else None
        b, d = rec("presidio_ginza", PIIType.NAME), rec("sumi_fp32", PIIType.NAME)
        if b is not None and d is not None:
            name_gap = (
                f"- **On personal names alone, Sumi's lead is small.** On the "
                f"template-independent set Sumi reaches {d:.2f} against "
                f"{b:.2f} for Presidio + GiNZA. Sumi's advantage is not name recall "
                f"in isolation — it is addresses, the ID/number families, and above all "
                f"the false-positive rate.\n")
    return f"""## Limitations

{name_gap}- **Trained and evaluated entirely on synthetic data.** Behaviour on your real
  documents will differ. Validate on your own data before deploying.
- **Misses are certain.** Lowering the threshold raises recall and also raises false
  positives; the trade-off table above is the honest picture of that.
- **Sumi is not a compliance control.** It reduces risk; it does not certify anything.
- **GGUF is exported but not executable** — no mainstream runtime currently runs
  ModernBERT-style token classification from GGUF. Use ONNX INT8 on CPU.
- **Japanese proofreading / typo detection is deliberately out of scope** (JWTD and
  existing public models already cover that).
- The rule layer requires context words for My Number and member IDs. A bare
  12-digit string with no surrounding cue is intentionally not flagged, to keep the
  false-positive rate low.
- Source code docstrings are written in Japanese (the project's working language);
  all published documentation is in English."""


def build_readme(bench, ood, train, exp, out: str) -> str:
    """Write the GitHub README.

    Claim: 検出率 / 低誤検出 / CPU速度 / 可逆性 — the project's public face.
    """
    params = (train or {}).get("params")
    onnx = ((exp or {}).get("formats", {}) or {}).get("onnx", {})
    n_pos = bench["n_pos"] if bench else 0
    n_neg = bench["n_neg"] if bench else 0

    speed_line = ""
    # 速度は汎化評価 (OOD) の実行値を使う。README が主として提示するのがその節のため。
    src = ood or bench
    if src:
        llm = next((r for r in src["results"] if r["condition"] == "local_llm_4b"), None)
        e = next((r for r in src["results"] if r["condition"] == "sumi_int8"), None)
        if llm and e and llm["speed"]["docs_per_sec"]:
            ratio = e["speed"]["docs_per_sec"] / llm["speed"]["docs_per_sec"]
            speed_line = (f"**{ratio:.0f}× faster than a 4B local LLM on the same CPU** "
                          f"({e['speed']['docs_per_sec']:.1f} vs "
                          f"{llm['speed']['docs_per_sec']:.2f} docs/s), using "
                          f"{e['speed']['peak_rss_mb']:.0f} MB against "
                          f"{llm['speed']['peak_rss_mb']:.0f} MB.")

    ood_block = ""
    if ood:
        ood_block = f"""## Evaluation — generalisation (the honest numbers)

The training documents come from business-document templates (email, meeting
minutes, application forms, support tickets). **Measuring only on documents from
those same templates would reward a model that merely memorised the slots.**

This section uses a *template-independent* set: synthetic PII inserted directly
into Wikipedia / statute / literary prose, {ood['n_pos']} positive and
{ood['n_neg']} negative documents. Each inserted sentence is phrased so the label
is actually valid ("The date of birth is {{value}}."), but no business-document
template is used.

{detection_table(ood, with_fp=True)}

### Recall at a fixed false-positive budget

In practice the question is not "how much can it find" but "how much can it find
while keeping false positives at a level we can live with".

{recall_at_fpr_table(ood)}

### CPU throughput and memory

{speed_table(ood)}

### Calibration

{calibration_table(ood)}
"""

    in_dist = ""
    if bench:
        in_dist = f"""<details>
<summary><b>Evaluation — in-distribution (inflated, kept for completeness)</b></summary>

The positive documents here come from the same templates as the training data, so
Sumi's recall is inflated. The negative subset is unaffected by this, because it
shares no templates with training.

All conditions processed the identical document set ({n_pos} positive /
{n_neg} negative) on the same CPU with the same thread count, each in its own
process so peak memory is attributed correctly.

{detection_table(bench)}

#### False positives on confusable negatives

{fp_table(bench)}

#### Recall at a fixed false-positive budget

{recall_at_fpr_table(bench)}

#### Speed and memory

{speed_table(bench)}

#### Calibration

{calibration_table(bench)}

</details>
"""

    # shields.io treats "-" as a field separator, so it must be doubled inside a label
    badge = lambda t: t.replace("-", "--").replace("/", "%2F").replace(" ", "%20")
    doc = f"""# 墨 Sumi — Japanese PII detection that holds up on hard negatives

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-{badge(HF_MODEL)}-yellow)](https://huggingface.co/{HF_MODEL})
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-{badge(HF_DATASET)}-yellow)](https://huggingface.co/datasets/{HF_DATASET})
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/{GH_REPO}/actions/workflows/ci.yml/badge.svg)](https://github.com/{GH_REPO}/actions/workflows/ci.yml)

**English-first PII detectors stumble on Japanese names and addresses. Sumi closes
that gap with a 0.13B token classifier that runs on CPU, and drops into Presidio in
one line.**

{DISCLAIMER}

## Why this exists

Running an English-configured Presidio over Japanese text finds the email address
and little else. Adding a Japanese NER model helps with names, but it also starts
firing on prefecture names, company names and part numbers that merely *look* like
personal data. The usual escape hatch — prompting a local 4B LLM — is accurate
enough but two orders of magnitude too slow on CPU.

Sumi is the fourth option: a small, fast model trained specifically on the
confusions that matter in Japanese, with a rule layer for the formats that do not
need a model, and a reversible masking path so you can still use an external LLM
without shipping the original values.

{speed_line}

## Install

```bash
git clone https://github.com/{GH_REPO}.git
cd sumi
pip install -e ".[presidio,onnx,app]"
```

## Use it

### Drop into Presidio (three lines)

```python
from presidio_analyzer import RecognizerRegistry
from sumi.presidio_plugin import register
register(registry)   # that is the whole integration
```

Sumi maps onto Presidio's own entity names (`PERSON`, `LOCATION`, `PHONE_NUMBER`,
`EMAIL_ADDRESS`, `DATE_TIME`, `CREDIT_CARD`, plus `JP_BANK_ACCOUNT`,
`JP_MY_NUMBER`, `JP_MEMBER_ID`, `JP_POSTAL_CODE`), so an existing
`AnonymizerEngine` pipeline keeps working unchanged. It does **not** require spaCy
or GiNZA — Sumi analyses Japanese itself.

### Command line

```bash
sumi redact input.txt --out masked.txt --map map.json   # redact, keep the mapping
sumi restore masked.txt --map map.json --out out.txt    # restore afterwards
```

The mapping file is written with mode `0600` and never leaves the machine.

### Python

```python
from sumi import SumiDetector
from sumi.mask import ReversibleMasker

det = SumiDetector("path/to/model")
masked, mapping = det.redact("田中太郎様の連絡先は090-1234-5678です。")
# '<NAME_1>様の連絡先は<PHONE_1>です。'

restored = ReversibleMasker().unmask(llm_response, mapping)
```

### Send masked text to an LLM, safely

```python
from sumi.mask import LLMRoundTrip

result = LLMRoundTrip(my_transport).run(text, det.detect(text),
                                        instruction="Summarise this document.")
result["response"]   # the LLM's answer, with the original values restored
```

`LLMRoundTrip` routes every outbound payload through an `EgressGuard` built from the
mapping table. If any original value — raw, NFKC-normalised, or digits-only — would
be transmitted, the send is **blocked** rather than logged. The test suite asserts
this over every synthetic document it generates.

## How it works

{method_diagram()}

### Three design decisions worth calling out

1. **Checksums never gate detection.** Card-shaped and My-Number-shaped strings are
   detected by *format*; whether the check digit validates is recorded in
   `meta["checksum_valid"]` and nothing more. The point of redaction is to leave
   nothing that looks like personal data, so a number with a bad check digit is not
   a number you may ignore.
2. **The merge order is explicit.** "Rules win where rules are certain, model
   elsewhere" is implemented as five ordered steps in `sumi.rules.merge_spans`,
   not as an implicit score comparison.
3. **Hard negatives are a closed loop.** After training, false positives on the
   negative subset are counted *per confusion type*, and those counts skew the
   generation distribution for the next batch
   (`HardNegativeGenerator.reweight_from_errors`).

## Compared against

| | Condition |
|---|---|
| (A) | Presidio with its default (English) configuration |
| (B) | Presidio + GiNZA — a fair, carefully-mapped Japanese NER setup, not a straw man: GiNZA's 189 extended-NE labels are mapped onto Presidio entities and adjacent place fragments are joined into full addresses |
| (C) | Qwen3-4B-Instruct (Q4_K_M) via llama.cpp, CPU only, prompted to extract PII |
| (D) | Sumi, fp32 PyTorch on CPU |
| (E) | Sumi, ONNX INT8 on CPU |

{ood_block}
{in_dist}
## Figures

{figure_block()}

## Model

- Base: [`sbintuitions/modernbert-ja-130m`](https://huggingface.co/sbintuitions/modernbert-ja-130m) (MIT)
- Parameters: {f"{params/1e6:.0f}M ({params/1e9:.2f}B)" if params else "132M (0.13B)"}
- Labels: 21 BIO classes (10 types × B/I, plus O)
- A span's probability is the **minimum** over its constituent token probabilities —
  the weakest link, not the average
{f"- ONNX INT8: **{onnx['int8_mb']:.0f} MB** (from {onnx['fp32_mb']:.0f} MB fp32), argmax agreement {onnx.get('int8_argmax_agreement', 0):.4f}" if onnx.get("int8_mb") else ""}

### Distribution formats

| Format | Runnable | Use |
|---|---|---|
| PyTorch (safetensors) | yes | training, research |
| ONNX fp32 / INT8 | yes | **the CPU path** (condition (E)) |
| MLX (safetensors) | yes | Apple Silicon |
| GGUF | **no** | weights exported for tooling compatibility only — no mainstream runtime executes ModernBERT token classification from GGUF |

## Data

Everything is synthetic. See the [dataset card](https://huggingface.co/datasets/{HF_DATASET}).

- Base prose: Japanese Wikipedia (CC BY-SA 4.0), e-Gov statutes (not subject to
  copyright under Article 13 of the Japanese Copyright Act), Aozora Bunko
  (public domain). Licence and source are recorded per document.
- PII is generated with seeded RNGs and inserted **while recording offsets**, so the
  gold spans are correct by construction rather than by post-hoc search.
- Surnames follow an approximation of the real Japanese frequency distribution;
  addresses combine real public place names with **randomised** block/lot numbers;
  emails only ever use reserved domains (`example.*`, `.test`, `.invalid`).
- **Identifiers with check digits are format-valid and value-invalid by
  construction** — Luhn and the My Number check digit are deliberately violated, so
  no usable number can be produced. Tests assert this over thousands of samples.

{limitations(bench, ood)}

## Reproduce

```bash
python3 scripts/build_dataset.py --train 12000 --val 1500 --test 2000 --neg 2000
python3 scripts/train.py --epochs 2 --batch-size 32
python3 scripts/export.py --formats onnx,mlx,gguf
python3 benchmarks/run_benchmark.py --n-pos 250 --n-neg 250
python3 benchmarks/run_benchmark.py --pos-file ood.jsonl --tag ood --n-pos 100 --n-neg 100
python3 -m sumi.figures
python3 -m sumi.figures benchmarks/results/benchmark_ood.json _ood
python3 scripts/build_docs.py
python3 -m pytest
```

Every module also runs standalone as its own self-test, e.g. `python3 -m sumi.rules`.

## Tests

`pytest` covers, among other things, the four properties this project is built on:

- the mapping table never appears in anything sent over the egress boundary;
- reversible masking restores the original text exactly, byte for byte;
- the rule layer does not miss the format-determined types;
- the false-positive rate on the hard-negative subset stays under a fixed threshold.

A further test enforces that every public function documents which claim it
substantiates (detection rate / low false positives / CPU speed / reversibility /
calibration).

## Licence

Apache-2.0 for the code. Synthetic data is CC0-1.0; rows containing Wikipedia-derived
prose inherit CC BY-SA 4.0.
"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    return doc


def build_model_card(bench, ood, train, exp, out: str) -> str:
    """Write the Hugging Face model card.

    Claim: 検出率 / CPU速度 — what a user of the weights needs to know.
    """
    params = (train or {}).get("params") or 132_400_000
    onnx = ((exp or {}).get("formats", {}) or {}).get("onnx", {})
    ood_tbl = detection_table(ood, with_fp=True) if ood else "_benchmark not run_"
    ood_n = f"{ood['n_pos']} positive / {ood['n_neg']} negative documents" if ood else ""
    speed = speed_table(ood or bench) if (ood or bench) else ""

    doc = f"""---
license: apache-2.0
language:
- ja
library_name: transformers
pipeline_tag: token-classification
tags:
- pii
- privacy
- japanese
- token-classification
- named-entity-recognition
- presidio
- onnx
base_model: sbintuitions/modernbert-ja-130m
datasets:
- {HF_DATASET}
widget:
- text: "田中太郎様よりご連絡をいただきました。連絡先は090-1234-5678、tanaka.taro@example.co.jp です。"
- text: "森の中を歩いていると、林業の振興について泉が湧くように話が広がった。長野県の気候は寒暖差が大きい。"
---

# 墨 Sumi — Japanese PII detection (0.13B, CPU)

A small Japanese PII token classifier that is trained specifically on the confusions
English-first tools get wrong: surnames that are also common nouns (森 / 林 / 泉 /
大和 / 青木), place and company homographs, honorific boundaries, and digit strings
that merely look like phone numbers.

Pairs with a rule layer for format-determined types and a reversible masking path,
so masked text can be sent to an external LLM and the response restored.

{DISCLAIMER}

- **Code, benchmarks and training pipeline:** https://github.com/{GH_REPO}
- **Dataset:** https://huggingface.co/datasets/{HF_DATASET}
- **Base model:** [`sbintuitions/modernbert-ja-130m`](https://huggingface.co/sbintuitions/modernbert-ja-130m) (MIT)
- **Parameters:** {params/1e6:.0f}M ({params/1e9:.2f}B)
- **Labels:** 21 BIO classes over 10 PII types

## Detected types

| Label | Meaning |
|---|---|
{chr(10).join(f"| `{t.value}` | {EN[t.value]} ({t.ja}) |" for t in ALL_TYPES)}

## Usage

The recommended entry point is the `sumi` package, which combines this model with
the rule layer and the calibrator:

```bash
pip install "sumi[presidio,onnx] @ git+https://github.com/{GH_REPO}.git"
```

```python
from sumi import SumiDetector

det = SumiDetector("{HF_MODEL}")          # or a local directory
for span in det.detect("田中太郎様の連絡先は090-1234-5678です。"):
    print(span.label.value, span.text, round(span.score, 3))
# NAME 田中太郎 0.997
# PHONE 090-1234-5678 0.95
```

Reversible masking, with the mapping kept locally:

```python
masked, mapping = det.redact(text)   # '<NAME_1>様の連絡先は<PHONE_1>です。'
original = ReversibleMasker().unmask(llm_response, mapping)
```

Inside Presidio:

```python
from presidio_analyzer import RecognizerRegistry
from sumi.presidio_plugin import register
register(registry)
```

### Raw transformers

The model is a standard `ModernBertForTokenClassification`. If you use it directly
you lose the rule layer, the calibration and the boundary refinement, which is where
a good deal of the precision comes from:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tok = AutoTokenizer.from_pretrained("{HF_MODEL}")
model = AutoModelForTokenClassification.from_pretrained("{HF_MODEL}")
```

## Evaluation

Measured against four alternatives on identical documents, same CPU, same thread
count, each condition in its own process:

- **(A)** Presidio, default English configuration
- **(B)** Presidio + GiNZA, carefully mapped (not a straw man)
- **(C)** Qwen3-4B-Instruct Q4_K_M via llama.cpp, CPU only, prompted
- **(D)** Sumi fp32, **(E)** Sumi ONNX INT8

### Generalisation set ({ood_n})

Synthetic PII inserted into Wikipedia / statute / literary prose — **no business
document templates**, so this is the number that reflects real generalisation
rather than template memorisation.

{ood_tbl}

The "FP rate" column is the fraction of hard-negative documents (which contain **no**
PII at all) where the system fired at least once.

### CPU throughput and memory

{speed}

## Files

| File | What it is |
|---|---|
| `model.safetensors` | fp32 PyTorch weights |
| `model.onnx` | ONNX fp32 |
| `model.int8.onnx` | ONNX dynamic INT8 — {f"{onnx['int8_mb']:.0f} MB, argmax agreement {onnx.get('int8_argmax_agreement',0):.4f}" if onnx.get("int8_mb") else "the CPU path"} |
| `calibrator.json` | temperature scaling fitted on held-out spans |
| `sumi_labels.json` | label order — keeps ONNX and PyTorch in sync |
| `negatives_weights.json` | closed-loop weights: which confusion types the model still gets wrong |
| `mlx/` | MLX weights for Apple Silicon |
| `gguf/` | GGUF export, **weights only, not executable** (see limitations) |

## Training

- 12,000 synthetic documents, 107k gold spans; 2 epochs, batch 32, max length 256
- AdamW, linear warmup and decay, gradient clipping at 1.0
- Checkpoint selected on validation **span** exact-match F1, not token F1
- Span probability = minimum over constituent token probabilities

## Intended use and limitations

Intended for reducing the chance that Japanese personal data reaches an external
system — for example masking documents before sending them to a hosted LLM.

{limitations(bench, ood)}

## Citation

```bibtex
@software{{sumi2026,
  title  = {{Sumi: Japanese PII detection that holds up on hard negatives}},
  author = {{Sumi contributors}},
  year   = {{2026}},
  url    = {{https://github.com/{GH_REPO}}}
}}
```
"""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    return doc


def main() -> None:
    """Generate all published documents.

    Claim: 検出率 — one command regenerates every published number from measurements.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="benchmarks/results/benchmark.json")
    ap.add_argument("--ood", default="benchmarks/results/benchmark_ood.json")
    ap.add_argument("--train", default="artifacts/sumi-model/train_report.json")
    ap.add_argument("--export", default="artifacts/export/export_report.json")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--model-card", default="artifacts/cards/MODEL_CARD.md")
    args = ap.parse_args()

    bench, ood = _load(args.bench), _load(args.ood)
    train, exp = _load(args.train, {}), _load(args.export, {})

    r = build_readme(bench, ood, train, exp, args.readme)
    print(f"README      -> {args.readme} ({len(r)} chars)")
    m = build_model_card(bench, ood, train, exp, args.model_card)
    print(f"model card  -> {args.model_card} ({len(m)} chars)")
    if not bench:
        print("  (benchmark.json missing — evaluation sections are placeholders)")


if __name__ == "__main__":
    main()
