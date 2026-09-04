"""Emit the rule layer as JSON so the browser demo runs the same rules as Python.

Claim: 低誤検出 — the static Space has to re-implement the rule layer in
JavaScript. Transcribing the patterns by hand would let the demo and the library
drift apart silently. Instead the patterns, digit rules, context words and merge
precedence are exported from the Python definitions and interpreted by the JS.

    python3 scripts/export_rules_json.py --out space/rules.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumi.rules import DEFAULT_SPECS, _AREA2, _AREA3, _IP, _MOBILE, _TOLLFREE4
from sumi.types import MODEL_DRIVEN, RULE_DETERMINISTIC, ALL_TYPES


def export() -> dict:
    """Serialise everything the JS rule layer needs.

    Claim: 低誤検出 — one source of truth for the regexes and the JP numbering plan.
    """
    specs = []
    for s in DEFAULT_SPECS:
        specs.append({
            "rule_id": s.rule_id,
            "label": s.label.value,
            "pattern": s.pattern,
            "confidence": s.confidence,
            "require_context": s.require_context,
            "context": list(s.context),
            "negative_context": list(s.negative_context),
            "priority": s.priority,
            # The JS side re-implements these by name; Python functions cannot travel.
            "validator": ("jp_phone" if s.validator is not None else None),
            "reject_date_shape": s.label.value == "PHONE",
        })
    return {
        "schema": "sumi-rules-v1",
        "specs": specs,
        "phone_plan": {
            "area2": sorted(_AREA2),
            "area3": sorted(_AREA3),
            "mobile": sorted(_MOBILE),
            "ip": sorted(_IP),
            "tollfree4": sorted(_TOLLFREE4),
        },
        "types": {
            t.value: {"ja": t.ja, "en": t.en,
                      "rule_deterministic": t in RULE_DETERMINISTIC,
                      "model_driven": t in MODEL_DRIVEN}
            for t in ALL_TYPES
        },
        "rule_deterministic": [t.value for t in RULE_DETERMINISTIC],
        "context_window": 12,
    }


def main() -> None:
    """Write the JSON bundle.

    Claim: 低誤検出 — regenerate this whenever the rules change.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="space/rules.json")
    args = ap.parse_args()
    data = export()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"{len(data['specs'])} rules -> {args.out} "
          f"({os.path.getsize(args.out)/1024:.1f} KB)")
    for s in data["specs"]:
        print(f"  {s['rule_id']:28s} {s['label']:13s} "
              f"ctx={'required' if s['require_context'] else 'bonus':8s} "
              f"neg={len(s['negative_context'])}")


if __name__ == "__main__":
    main()
