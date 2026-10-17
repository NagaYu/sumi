"""Measure how closely the INT8 ONNX model agrees with fp32, on real evaluation text.

Claim: CPU速度 / 検出率 — scripts/export.py reports argmax agreement on a single
sanity-check sentence, where it is 1.0000. That number is true but says little:
on the benchmark, INT8 does not label every document identically to fp32
(compare conditions (D) and (E)). This script measures agreement over the
evaluation sets so the published figure has an honest scope.

    python3 scripts/measure_int8_agreement.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main() -> None:
    """Run fp32 PyTorch and INT8 ONNX side by side and compare argmax labels.

    Claim: CPU速度 — the quantised model's fidelity, measured where it matters.
    """
    import onnxruntime as ort
    import torch

    from scripts.build_dataset import read_jsonl
    from sumi.model import TokenClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/sumi-model")
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--out", default="benchmarks/results/int8_agreement.json")
    ap.add_argument("--n-ood", type=int, default=100)
    ap.add_argument("--n-neg", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=250)
    args = ap.parse_args()

    torch.set_num_threads(8)
    clf = TokenClassifier.load(args.model, device="cpu")
    sess = ort.InferenceSession(os.path.join(args.model, "model.int8.onnx"),
                                providers=["CPUExecutionProvider"])
    docs = (read_jsonl(os.path.join(args.data, "ood.jsonl"))[: args.n_ood]
            + read_jsonl(os.path.join(args.data, "negatives.jsonl"))[: args.n_neg]
            + read_jsonl(os.path.join(args.data, "test.jsonl"))[: args.n_test])

    agree = total = pii_agree = pii_total = identical = 0
    for d in docs:
        enc = clf.tokenizer(d.text, truncation=True, max_length=256, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        am = enc["attention_mask"].astype(np.int64)
        with torch.no_grad():
            a = clf.model(input_ids=torch.from_numpy(ids),
                          attention_mask=torch.from_numpy(am)).logits.numpy()[0].argmax(-1)
        b = sess.run(None, {"input_ids": ids, "attention_mask": am})[0][0].argmax(-1)
        a, b = a[1:-1], b[1:-1]                      # drop special tokens
        agree += int((a == b).sum())
        total += len(a)
        m = (a != 0) | (b != 0)                      # either model says "PII"
        pii_agree += int((a[m] == b[m]).sum())
        pii_total += int(m.sum())
        identical += int((a == b).all())

    result = {
        "documents": len(docs),
        "composition": {"ood": args.n_ood, "negatives": args.n_neg, "test": args.n_test},
        "max_length": 256,
        "token_agreement": agree / total,
        "tokens": total,
        "pii_token_agreement": pii_agree / pii_total,
        "pii_tokens": pii_total,
        "documents_identical": identical,
        "note": "argmax over 21 BIO labels, fp32 PyTorch vs INT8 ONNX, special tokens excluded",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
