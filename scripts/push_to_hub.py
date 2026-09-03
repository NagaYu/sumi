"""学習済みモデル・書き出し成果物・カードを Hugging Face Hub へ公開する。

Claim: 検出率 / CPU速度 — 評価に使ったものと同一の成果物を公開し、
第三者が同じ数値を再現できるようにする。

**公開は明示的な操作** であり、既定では何もしない。
実行にはリポジトリ ID の明示と ``--yes`` の両方が必要。

    python3 scripts/push_to_hub.py --repo <user>/sumi-ja-pii --yes
    python3 scripts/push_to_hub.py --repo <user>/sumi-ja-pii --dry-run   # 中身の確認だけ
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Files to publish with the model. Anything not matched here is skipped.
INCLUDE = (
    "config.json", "model.safetensors", "sumi_labels.json", "calibrator.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "spiece.model", "vocab.txt", "merges.txt", "train_report.json",
    "negatives_weights.json", "model.onnx", "model.int8.onnx", "README.md",
)

#: Exact filenames that must never be published.
FORBIDDEN_EXACT = frozenset({
    "map.json", "mapping.json", ".env", ".netrc", "token", "credentials.json",
    "secrets.json", "id_rsa", "id_ed25519",
})

#: Substrings that mark a file as sensitive. Checked only after the tokenizer
#: allowlist below, because "tokenizer_config.json" contains "token".
FORBIDDEN_SUBSTRINGS = ("secret", "credential", "password", "apikey", "api_key",
                        "private_key", "mapping_table")

#: Legitimate model files whose names collide with the substrings above.
ALWAYS_ALLOWED = frozenset({
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "tokenizer.model",
})


def is_sensitive(name: str) -> tuple[bool, str]:
    """Decide whether a file must be withheld from publication.

    Claim: 可逆性 — the mapping table is the key that reverses redaction, so the
    publish path itself refuses to upload it.

    The check is deliberately precise rather than a broad substring match: an
    earlier version matched "token" as a substring and silently excluded
    ``tokenizer_config.json``, which would have shipped an unusable model.
    """
    low = name.lower()
    if low in ALWAYS_ALLOWED:
        return False, ""
    if low in FORBIDDEN_EXACT:
        return True, "exact match on a forbidden filename"
    for frag in FORBIDDEN_SUBSTRINGS:
        if frag in low:
            return True, f"contains {frag!r}"
    # A mapping table produced by `sumi redact` — never publishable.
    if low.endswith(".json") and low.startswith("map"):
        return True, "looks like a redaction mapping table"
    return False, ""


def collect(model_dir: str) -> list[str]:
    """List the files to publish, refusing anything sensitive.

    Claim: 可逆性 — publication must not be able to leak the mapping table.
    """
    out = []
    for name in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            continue
        bad, why = is_sensitive(name)
        if bad:
            print(f"  withheld ({why}): {name}")
            continue
        if name in INCLUDE or name.endswith((".safetensors", ".json", ".txt", ".model", ".onnx")):
            out.append(path)
    return out


def _selftest() -> None:
    """Check the sensitivity filter against the files it must and must not block.

    Claim: 可逆性 — a regression here either leaks the mapping table or ships a
    broken model, so it is worth asserting explicitly.
    """
    must_publish = ["config.json", "model.safetensors", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json",
                    "sumi_labels.json", "calibrator.json", "model.int8.onnx"]
    must_withhold = ["map.json", "mapping.json", ".env", "my_secret.json",
                     "aws_credentials.json", "api_key.txt", "mapping_table.json"]
    for n in must_publish:
        bad, why = is_sensitive(n)
        assert not bad, f"{n} would be withheld ({why}) — the model would be unusable"
    for n in must_withhold:
        bad, _ = is_sensitive(n)
        assert bad, f"{n} would be published"
    print(f"filter self-test: {len(must_publish)} publishable, "
          f"{len(must_withhold)} withheld — OK")


def main() -> None:
    """公開を実行する (``--yes`` 必須)。

    Claim: 検出率 — 公開物と評価物の同一性を保つ。
    """
    ap = argparse.ArgumentParser(description="Publish the Sumi model to Hugging Face")
    ap.add_argument("--repo", required=True, help="e.g. your-name/sumi-ja-pii")
    ap.add_argument("--model", default="artifacts/sumi-model")
    ap.add_argument("--card", default="artifacts/cards/MODEL_CARD.md",
                    help="model card with HF frontmatter")
    ap.add_argument("--export", default="artifacts/export", help="GGUF/MLX export directory")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list what would be published, then stop")
    ap.add_argument("--yes", action="store_true", help="actually publish (explicit consent)")
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        print(f"no model at {args.model}")
        raise SystemExit(1)

    files = collect(args.model)
    print("=" * 70)
    print(f"target: https://huggingface.co/{args.repo}")
    print("=" * 70)
    total = 0
    for p in files:
        sz = os.path.getsize(p)
        total += sz
        print(f"  {os.path.basename(p):32s} {sz/1e6:8.1f} MB")
    extra = []
    for sub in ("gguf", "mlx"):
        d = os.path.join(args.export, sub)
        if os.path.isdir(d):
            for n in sorted(os.listdir(d)):
                p = os.path.join(d, n)
                if os.path.isfile(p):
                    extra.append((f"{sub}/{n}", p))
                    total += os.path.getsize(p)
                    print(f"  {sub + '/' + n:32s} {os.path.getsize(p)/1e6:8.1f} MB")
    if os.path.exists(args.card):
        print(f"  {'README.md (model card)':32s} "
              f"{os.path.getsize(args.card)/1e3:8.1f} KB")
    print(f"  {'total':32s} {total/1e6:8.1f} MB")

    if not args.yes or args.dry_run:
        print("\nPass --yes to publish. Nothing is uploaded by default.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for p in files:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p),
                        repo_id=args.repo, repo_type="model")
        print(f"  up: {os.path.basename(p)}")
    for rel, p in extra:
        api.upload_file(path_or_fileobj=p, path_in_repo=rel,
                        repo_id=args.repo, repo_type="model")
        print(f"  up: {rel}")
    if os.path.exists(args.card):
        api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model")
        print("  up: README.md")
    print(f"\ndone: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
