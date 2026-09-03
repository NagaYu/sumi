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

#: モデルリポジトリに含めるファイル (対応表など機微なものは絶対に含めない)
INCLUDE = (
    "config.json", "model.safetensors", "sumi_labels.json", "calibrator.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "spiece.model", "vocab.txt", "train_report.json", "negatives_weights.json",
    "model.onnx", "model.int8.onnx",
)

#: 絶対に公開してはならないファイル名の断片
FORBIDDEN = ("map.json", "mapping", "secret", ".env", "token", "credential")


def collect(model_dir: str) -> list[str]:
    """公開対象ファイルを列挙し、機微なファイルを排除する。

    Claim: 可逆性 — 対応表 (map.json) のような復元の鍵が
    誤って公開されることを、公開経路の側でも防ぐ。
    """
    out = []
    for name in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, name)
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if any(f in low for f in FORBIDDEN):
            print(f"  除外 (機微): {name}")
            continue
        if name in INCLUDE or name.endswith((".safetensors", ".json", ".txt", ".model")):
            out.append(p)
    return out


def main() -> None:
    """公開を実行する (``--yes`` 必須)。

    Claim: 検出率 — 公開物と評価物の同一性を保つ。
    """
    ap = argparse.ArgumentParser(description="Sumi モデルを Hugging Face へ公開する")
    ap.add_argument("--repo", required=True, help="例: your-name/sumi-ja-pii")
    ap.add_argument("--model", default="artifacts/sumi-model")
    ap.add_argument("--card", default="README.md", help="モデルカード (HF frontmatter 付き)")
    ap.add_argument("--export", default="artifacts/export", help="GGUF/MLX 書き出し先")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="何が公開されるか表示するだけ")
    ap.add_argument("--yes", action="store_true", help="実際に公開する (明示的な同意)")
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        print(f"モデルがありません: {args.model}")
        raise SystemExit(1)

    files = collect(args.model)
    print("=" * 70)
    print(f"公開先: https://huggingface.co/{args.repo}")
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
        print(f"  {'README.md (モデルカード)':32s} "
              f"{os.path.getsize(args.card)/1e3:8.1f} KB")
    print(f"  {'合計':32s} {total/1e6:8.1f} MB")

    if not args.yes or args.dry_run:
        print("\n--yes を付けると実際に公開します (既定では公開しません)。")
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
    print(f"\n完了: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
