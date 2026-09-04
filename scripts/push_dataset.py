"""Publish the already-built synthetic corpus to the Hugging Face Hub.

Claim: 検出率 / 低誤検出 — publish exactly the data the benchmark was run on, so
a third party can reproduce the reported numbers rather than take them on trust.

Publication is an explicit action: nothing is uploaded without ``--yes``.

    python3 scripts/push_dataset.py --repo NagaYu/sumi-ja-pii-corpus --dry-run
    python3 scripts/push_dataset.py --repo NagaYu/sumi-ja-pii-corpus --yes
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_dataset import dataset_card, read_jsonl

SPLITS = ("train", "validation", "test", "ood", "negatives")


def load_splits(data_dir: str) -> dict[str, list]:
    """Read every split that exists on disk.

    Claim: 検出率 — publishes the on-disk corpus rather than regenerating it, so
    the uploaded rows are byte-identical to the ones the benchmark scored.
    """
    out = {}
    for name in SPLITS:
        path = os.path.join(data_dir, f"{name}.jsonl")
        if os.path.exists(path):
            out[name] = read_jsonl(path)
    return out


def to_rows(docs) -> list[dict]:
    """Flatten documents into Hub-friendly rows.

    Claim: 検出率 — ``spans`` is stored as a JSON string so that the offsets
    survive the round trip through Arrow without schema coercion surprises.
    """
    rows = []
    for d in docs:
        row = d.to_dict()
        row["spans"] = json.dumps(
            [s.to_dict() for s in d.sorted_spans()], ensure_ascii=False
        )
        row["meta"] = json.dumps(
            {k: v for k, v in (d.meta or {}).items() if not k.startswith("_")},
            ensure_ascii=False,
        )
        row["negative_kinds"] = list(d.negative_kinds or [])
        rows.append(row)
    return rows


def verify(sets: dict[str, list]) -> None:
    """Re-check the invariants that make the dataset safe to publish.

    Claim: 低誤検出 / 検出率 — refuses to publish if the negative split has gained
    gold spans, if any document's offsets no longer match its text, or if a
    check-digit identifier turned out to be valid (which would mean a usable
    number had been generated).
    """
    from sumi.rules import luhn_ok, mynumber_check_ok
    from sumi.types import PIIType

    for name, docs in sets.items():
        for d in docs:
            d.validate()
    if "negatives" in sets:
        bad = [d.doc_id for d in sets["negatives"] if d.spans]
        assert not bad, f"negative split has gold spans: {bad[:5]}"

    import re

    for name, docs in sets.items():
        for d in docs:
            for s in d.spans:
                digits = re.sub(r"\D", "", s.text)
                if s.label is PIIType.CREDIT_CARD:
                    assert not luhn_ok(digits), f"valid card number in {d.doc_id}"
                elif s.label is PIIType.MYNUMBER:
                    assert not mynumber_check_ok(digits), f"valid My Number in {d.doc_id}"
    print("  invariants OK: offsets valid, negatives empty, no usable identifiers")


def main() -> None:
    """Upload the corpus and its card.

    Claim: 検出率 — one command, and only with explicit consent.
    """
    ap = argparse.ArgumentParser(description="Publish the Sumi corpus to Hugging Face")
    ap.add_argument("--repo", required=True, help="e.g. your-name/sumi-ja-pii-corpus")
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--card", default="artifacts/cards/DATASET_CARD.md")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually publish")
    args = ap.parse_args()

    sets = load_splits(args.data)
    if not sets:
        print(f"no splits under {args.data}")
        raise SystemExit(1)

    print("=" * 70)
    print(f"target: https://huggingface.co/datasets/{args.repo}")
    print("=" * 70)
    for name, docs in sets.items():
        n_spans = sum(len(d.spans) for d in docs)
        size = os.path.getsize(os.path.join(args.data, f"{name}.jsonl")) / 1e6
        print(f"  {name:12s} {len(docs):6d} docs  {n_spans:7d} spans  {size:7.1f} MB")
    print("\nverifying invariants before upload ...")
    verify(sets)

    if not args.yes or args.dry_run:
        print("\nPass --yes to publish. Nothing is uploaded by default.")
        return

    from datasets import Dataset, DatasetDict, Features, Sequence, Value
    from huggingface_hub import HfApi

    # An explicit schema is required. Without it, Arrow infers the element type of
    # `negative_kinds` per split, and the `ood` split — where every list is empty —
    # comes out as list<null> instead of list<string>, so push_to_hub rejects the
    # DatasetDict for having mismatched features.
    features = Features({
        "doc_id": Value("string"),
        "text": Value("string"),
        "spans": Value("string"),            # JSON-encoded list of span objects
        "subset": Value("string"),
        "genre": Value("string"),
        "source_license": Value("string"),
        "source_ref": Value("string"),
        "negative_kinds": Sequence(Value("string")),
        "meta": Value("string"),             # JSON-encoded object
    })

    dd = DatasetDict({
        name: Dataset.from_list(to_rows(docs), features=features)
        for name, docs in sets.items()
    })
    schemas = {n: str(d.features) for n, d in dd.items()}
    assert len(set(schemas.values())) == 1, f"split schemas differ: {schemas}"
    print(f"  schema pinned across {len(dd)} splits")
    dd.push_to_hub(args.repo, private=args.private)

    card = (open(args.card, encoding="utf-8").read()
            if os.path.exists(args.card) else dataset_card(sets, args.repo))
    HfApi().upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md", repo_id=args.repo, repo_type="dataset",
    )
    print(f"\ndone: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
