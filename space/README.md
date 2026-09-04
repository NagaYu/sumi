---
title: Sumi — Japanese PII Detection
emoji: 🖋️
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Japanese PII detection running entirely in your browser
models:
  - NagaYu/sumi-ja-pii
datasets:
  - NagaYu/sumi-ja-pii-corpus
tags:
  - pii
  - privacy
  - japanese
  - presidio
  - transformers.js
---

# 墨 Sumi — Japanese PII detection, in your browser

Paste Japanese text and see what a 0.13B token classifier plus a rule layer finds.
The model is downloaded to your machine and runs locally with
[transformers.js](https://github.com/huggingface/transformers.js) — **nothing you
paste is uploaded anywhere.** For a redaction tool that seemed like the only honest
way to build a demo.

The second built-in example contains **no personal data at all** — only look-alikes:
surnames that are also ordinary nouns (森 forest, 林 woods, 泉 spring), prefecture
names that are also common surnames (長野, 福島), company names built from surnames,
and part numbers shaped like phone numbers. Sumi finds nothing there. Presidio +
GiNZA finds three things, none of them real.

## What you are looking at

- **Model:** [NagaYu/sumi-ja-pii](https://huggingface.co/NagaYu/sumi-ja-pii) — INT8
  ONNX, 133 MB, cached by your browser after the first visit
- **Code:** [github.com/NagaYu/sumi](https://github.com/NagaYu/sumi)
- **Dataset:** [NagaYu/sumi-ja-pii-corpus](https://huggingface.co/datasets/NagaYu/sumi-ja-pii-corpus)

The Presidio + GiNZA comparison is **pre-computed** for the three built-in examples,
because spaCy cannot run in a browser. Those numbers come from the same code path as
the published benchmark (`sumi.compare`), not from a separate hand-written setup.
Type your own text and you will see Sumi's output alone.

## Is the browser port the same as the library?

Yes, and it is checked rather than asserted. The rule patterns, the Japanese
numbering plan, the context words and the merge precedence are not re-typed in
JavaScript: they are read from `rules.json`, generated from the Python definitions
by `scripts/export_rules_json.py`. [`parity.html`](parity.html) runs the browser
detector against span-for-span reference output produced by the Python
`SumiDetector` on the same INT8 graph, and reports any difference.

Writing this port found a real bug in the Python library — the quantised model
would occasionally label a lone newline as a personal name, which inflated the
false-positive count. That is fixed upstream.

## Measured results

On a template-independent evaluation set (100 positive, 100 hard-negative
documents), against Presidio + GiNZA:

| | Sumi | Presidio + GiNZA |
|---|---|---|
| Documents with a false positive (no PII present) | **0%** | 83% |
| Address recall | **1.00** | 0.08 |
| Recall at a 5% false-positive budget | **0.98** | 0.01 |

Full methodology, including where Sumi's lead is small (personal names: 0.81 vs
0.78), is in the [repository README](https://github.com/NagaYu/sumi).

> **This does not guarantee legal or regulatory compliance.** It reduces the risk of
> personal data leaving your machine; it is not a complete detector, and misses will
> happen. All training and evaluation data is synthetic — no real personal
> information was used anywhere in this project.
