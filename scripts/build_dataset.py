"""合成PIIコーパスを生成し、JSONL として保存する (任意で HF Dataset へ push)。

Claim: 検出率 / 低誤検出 — 学習と評価の土台を、正解を構成的に作れる合成データで
用意する。否定例サブセットを分離することで「紛らわしい否定例での誤検出率」を
独立に測定できるようにする。

生成物 (data/dataset/):
    train.jsonl       学習用 (hard negative を一定割合で混入した陽性文書)
    validation.jsonl  検証用
    test.jsonl        評価用 (陽性)
    negatives.jsonl   否定例のみ。**正解スパン0件** — ここでの検出は全て誤検出
    dataset_card.md   ライセンスと合成である旨を明記したカード

使い方:
    python3 scripts/build_dataset.py --train 12000 --val 1500 --test 2000 --neg 2000
    python3 scripts/build_dataset.py --push <hf-user>/sumi-ja-pii   # 明示的な公開操作
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumi.corpus import license_table, load_base_corpus
from sumi.negatives import HardNegativeGenerator, attach_surface_index
from sumi.synth import build_documents
from sumi.types import ALL_TYPES, Document

OUT_DIR = "data/dataset"


def write_jsonl(docs: list[Document], path: str) -> None:
    """Document 列を JSONL へ書き出す。

    Claim: 検出率 — 正解スパンを機械可読な形で保存し、評価の再現性を担保する。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            d.validate()
            row = d.to_dict()
            row.pop("meta", None) if False else None
            row["meta"] = {k: v for k, v in (d.meta or {}).items() if not k.startswith("_")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[Document]:
    """JSONL から Document 列を読み戻す。

    Claim: 検出率 — 公開データセットからの評価再現を保証する。
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Document.from_dict(json.loads(line)))
    return out


def build(
    n_train: int, n_val: int, n_test: int, n_neg: int, *,
    seed: int = 0, n_base: int = 400, inject_ratio: float = 0.45,
    weights: dict[str, float] | None = None,
) -> dict[str, list[Document]]:
    """全サブセットを生成する。

    Claim: 検出率 / 低誤検出 — 陽性文書には hard negative を混ぜて
    「本物と紛らわしい非PIIが同居する」現実的な文脈を作り、
    否定例サブセットは正解0件に保って誤検出率の測定対象とする。

    Args:
        inject_ratio: 陽性文書のうち hard negative を混入する割合。
        weights: 否定例生成の型別重み (閉ループで更新されたものを渡せる)。
    """
    base = load_base_corpus(n_base, seed=seed)
    print(f"土台テキスト {len(base)} 件 " f"({dict(Counter(b.source for b in base))})")

    sets: dict[str, list[Document]] = {}
    for name, n, sd in [
        ("train", n_train, seed), ("validation", n_val, seed + 1), ("test", n_test, seed + 2)
    ]:
        docs = build_documents(n, seed=sd, base_items=base, subset=name)
        gen = HardNegativeGenerator(seed=sd + 500, weights=weights)
        out = []
        for i, d in enumerate(docs):
            if (i / max(1, len(docs))) < inject_ratio:
                d = gen.inject(d, k=gen.rng.randint(1, 3))
            out.append(d)
        attach_surface_index(out, gen)
        sets[name] = out
        print(f"  {name:11s} {len(out):6d} 文書 / 正解 {sum(len(d.spans) for d in out):6d} スパン "
              f"/ 否定例混入 {sum(1 for d in out if d.negative_kinds)} 文書")

    ood = build_ood_documents(max(200, n_test // 4), seed=seed + 700, base_items=base)
    sets["ood"] = ood
    print(f"  {'ood':11s} {len(ood):6d} 文書 / 正解 {sum(len(d.spans) for d in ood):6d} スパン "
          f"(テンプレート非依存: 地の文へ直接挿入)")

    ngen = HardNegativeGenerator(seed=seed + 900, weights=weights)
    negs = ngen.build_negative_documents(n_neg, base_items=base)
    attach_surface_index(negs, ngen)
    sets["negatives"] = negs
    assert all(len(d.spans) == 0 for d in negs), "否定例に正解スパンがある"
    print(f"  {'negatives':11s} {len(negs):6d} 文書 / 正解 0 スパン (ここでの検出は全て誤検出)")
    return sets


def build_ood_documents(
    n: int, *, seed: int = 0, base_items=None, subset: str = "ood",
) -> list[Document]:
    """テンプレートを使わず、公開文章の地の文へ直接 PII を挿入した評価集合を作る。

    Claim: 検出率 — 学習データは業務文書テンプレートから生成しているため、
    同じテンプレート由来の test で測ると「テンプレートの穴埋めを覚えただけ」でも
    高い数値が出てしまう。本関数は **テンプレートを一切使わず**、
    Wikipedia / 法令 / 文学作品の地の文に PII を挿入することで、
    文脈からの汎化を測る out-of-distribution な評価集合を作る。

    挿入は文境界で行い、開始位置を記録しながら文字列を組み立てるため、
    正解スパンは構成的に正しい (``Document.validate()`` を通す)。

    重要: 挿入する1文の枠は **種別ごとに用意する**。枠が型の意味を成立させないと
    正解ラベルが不当になるため (例:「1943年12月29日へ通知した」を生年月日として
    採点してはならない。文脈依存型の DOB / MYNUMBER / MEMBER_ID は、
    枠がその型を示していて初めて検出可能であるべき)。
    枠は素の1文であり業務文書テンプレートではないので、テンプレート非依存性は保たれる。
    """
    import random as _random

    from sumi.synth import PIIFactory
    from sumi.types import ALL_TYPES, PIIType, Span, normalize

    rng = _random.Random(seed)
    factory = PIIFactory(seed=seed + 31)
    bases = list(base_items or [])
    if not bases:
        raise ValueError("build_ood_documents には土台テキストが必要です")

    # 種別ごとに「その型として読める」1文の枠を用意する。
    # 枠は業務文書テンプレートではなく素の1文なので、テンプレート非依存性は保たれる。
    # 一方で、枠が型の意味を成立させないと正解ラベルが不当になる
    # (例: 「1943年12月29日へ通知した」を生年月日として採点するのは誤り)。
    FRAMES: dict[PIIType, list[str]] = {
        PIIType.NAME: [
            "連絡先の担当は{v}である。", "{v}が窓口として記録されている。",
            "本件の申請者は{v}であった。", "名簿には{v}と記載されている。",
            "{v}宛に通知が送られた。",
        ],
        PIIType.ADDRESS: [
            "住所は{v}である。", "送付先は{v}と記録されている。",
            "現住所として{v}が届け出られた。", "居住地は{v}であった。",
        ],
        PIIType.PHONE: [
            "電話番号は{v}である。", "連絡先の電話は{v}と記録されている。",
            "TEL {v} まで問い合わせること。", "携帯番号として{v}が登録された。",
        ],
        PIIType.EMAIL: [
            "メールアドレスは{v}である。", "連絡用のメールは{v}と記録されている。",
            "電子メール {v} 宛に送付した。",
        ],
        PIIType.DOB: [
            "生年月日は{v}である。", "{v}生まれと記録されている。",
            "本人の生年月日として{v}が届け出られた。", "誕生日は{v}であった。",
        ],
        PIIType.BANK_ACCOUNT: [
            "振込先の口座は{v}である。", "引落口座として{v}が登録された。",
            "口座番号は{v}と記録されている。",
        ],
        PIIType.CREDIT_CARD: [
            "カード番号は{v}である。", "決済用のクレジットカード番号として{v}が登録された。",
        ],
        PIIType.MYNUMBER: [
            "マイナンバーは{v}である。", "個人番号として{v}が届け出られた。",
            "本人確認のため個人番号 {v} を確認した。",
        ],
        PIIType.MEMBER_ID: [
            "会員番号は{v}である。", "顧客番号として{v}が記録されている。",
            "契約番号は{v}であった。",
        ],
        PIIType.POSTAL_CODE: [
            "郵便番号は{v}である。", "〒{v} が届け出られている。",
            "住所の郵便番号として{v}が記録された。",
        ],
    }

    docs: list[Document] = []
    for i in range(n):
        item = rng.choice(bases)
        prose = normalize(getattr(item, "text", str(item)))
        sents = [x for x in prose.replace("\n", "。").split("。") if x.strip()]
        if len(sents) < 2:
            sents = [prose]
        n_pii = rng.randint(2, 6)
        types = [rng.choice(ALL_TYPES) for _ in range(n_pii)]

        parts: list[str] = []
        spans: list[Span] = []
        pos = 0
        # 文と「PII を含む1文」を交互に並べる
        order = list(sents[: max(2, n_pii + 2)])
        pii_at = sorted(rng.sample(range(len(order) + 1), min(n_pii, len(order) + 1)))
        pi = 0
        for idx in range(len(order) + 1):
            while pi < len(pii_at) and pii_at[pi] == idx:
                t = types[pi % len(types)]
                val = factory.make(t)
                frame = rng.choice(FRAMES[t])
                pre, post = frame.split("{v}")
                parts.append(pre)
                pos += len(pre)
                start = pos
                parts.append(val.text)
                pos += len(val.text)
                spans.append(Span(start, pos, t, val.text, 1.0,
                                  meta={"synthetic": True}))
                parts.append(post)
                pos += len(post)
                pi += 1
            if idx < len(order):
                seg = order[idx].strip() + "。"
                parts.append(seg)
                pos += len(seg)

        text = "".join(parts)
        # 正規化で長さが変わるとオフセットが壊れるため、変わらないことを確認する
        if normalize(text) != text:
            continue
        d = Document(
            text=text, spans=spans, doc_id=f"ood-{i:05d}", subset=subset,
            genre="ood_prose",
            source_license=getattr(item, "license", "unknown"),
            source_ref=getattr(item, "source", ""),
            negative_kinds=[],
            meta={"note": "テンプレート非依存 (地の文へ直接挿入)"},
        )
        d.validate()
        docs.append(d)
    return docs


def dataset_card(sets: dict[str, list[Document]], repo_id: str = "") -> str:
    """データセットカード (ライセンス・合成である旨・内訳) を生成する。

    Claim: 検出率 — 「何で測ったか」を公開物に添付し、数値の解釈を可能にする。
    """
    type_counts = Counter(
        s.label.value for name in ("train", "validation", "test")
        for d in sets.get(name, []) for s in d.spans
    )
    kind_counts = Counter(
        k for docs in sets.values() for d in docs for k in d.negative_kinds
    )
    lic_rows = "\n".join(
        f"| {r['source']} | {r['license']} | {r.get('url','')} | {r.get('note','')} |"
        for r in license_table()
    )
    type_rows = "\n".join(
        f"| {t.value} | {t.ja} | {type_counts.get(t.value,0)} |" for t in ALL_TYPES
    )
    kind_rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(kind_counts.items()))
    split_rows = "\n".join(
        f"| {k} | {len(v)} | {sum(len(d.spans) for d in v)} |" for k, v in sets.items()
    )

    return f"""---
license: cc-by-sa-4.0
language:
- ja
task_categories:
- token-classification
tags:
- pii
- privacy
- japanese
- synthetic
- named-entity-recognition
size_categories:
- 10K<n<100K
---

# Sumi — synthetic Japanese PII corpus

**This dataset contains no real personal information.** Every name, address, phone
number, email address, date of birth, bank account, member ID and My-Number-shaped
digit string is generated with a seeded RNG and inserted into the text **while
recording the offsets**, so the gold spans are correct by construction rather than
recovered by searching afterwards.

Built for [Sumi](https://github.com/NagaYu/sumi), a Japanese PII detector.

## Why this dataset exists

Japanese PII detection fails in a specific way: surnames that are also ordinary
nouns (森 "forest", 林 "woods", 泉 "spring", 大和, 青木), place names that are also
surnames (長野, 福島, 千葉, 山口), company names built from surnames, honorific
boundaries (様 / さん / 氏 / 殿, and the many uses of 様 that are not personal at
all), part numbers that look like phone numbers, and facility names that look like
addresses. This corpus makes those confusions the point rather than an accident.

## Important notes

- **No real personal data.** Surnames are drawn from an approximation of the public
  frequency distribution; addresses combine real public place names with
  **randomised** block/lot numbers; email addresses only ever use reserved domains
  (`example.com`, `example.co.jp`, `example.ne.jp`, `.test`, `.invalid`, …).
- **Identifiers with check digits are format-valid and value-invalid by
  construction.** Card-shaped numbers deliberately fail Luhn and My-Number-shaped
  numbers deliberately fail the official check-digit algorithm, so no usable number
  can be produced. The project's test suite asserts this over thousands of samples.
- This dataset **does not certify compliance** with any law or regulation. It is
  material for training and evaluating detectors, and makes no completeness claim.

## Splits

| split | documents | gold spans |
|---|---:|---:|
{split_rows}

`negatives` contains **zero gold spans**. Anything a detector finds there is by
definition a false positive, which is what makes the false-positive rate measurable.

`ood` is the **template-independent** evaluation split. `train` / `validation` /
`test` are generated from business-document templates (email, meeting minutes,
application forms, support tickets), so measuring only on `test` would reward a
model that merely memorised the template slots. `ood` inserts the same synthetic PII
directly into Wikipedia / statute / literary prose, with each carrier sentence
phrased so the label is genuinely valid. **Report both.**

## Gold spans by type (train + validation + test)

| type | Japanese | count |
|---|---|---:|
{type_rows}

## Hard-negative kinds

Deliberately confusable material, mixed into the positive documents and forming the
entire `negatives` split.

| kind | count |
|---|---:|
{kind_rows}

- `common_noun_surname` — surnames that are also ordinary nouns (森 / 林 / 泉 / 大和 / 青木)
- `place_as_person` — prefecture and city names that are also common surnames, used as places
- `company_as_person` — company names built from surname morphemes
- `honorific_boundary` — 様 / さん / 氏 / 殿 with and without a person, including
  お客様, 皆様, 仕様, 同様, 様式 where 様 is not a person at all
- `phone_like_id` — part numbers, order numbers, ISBNs and dates shaped like phone numbers
- `address_like_facility` — civic buildings and stations that read like addresses
- `number_like_id` — bill numbers, statute numbers, quantities that look like account numbers
- `date_like_nondob` — contract, issue and meeting dates that are not birth dates

## Base text licences

Synthetic PII is inserted into openly-licensed Japanese prose. Each row records its
source and licence in `source_license` / `source_ref`.

| source | licence | url | note |
|---|---|---|---|
{lic_rows}

The synthetic parts (templates, PII values, hard negatives) are CC0-1.0. Rows
containing Wikipedia-derived prose inherit CC BY-SA 4.0, which is why the dataset as
a whole is published under CC BY-SA 4.0.

## Fields

- `doc_id` (string), `text` (string)
- `spans`: JSON-encoded list of `{{start, end, label, text, score, source, meta}}`.
  `start` / `end` are **Python character indices into the NFKC-normalised text**,
  half-open `[start, end)` — not byte offsets.
- `subset`, `genre`, `source_license`, `source_ref`, `negative_kinds`, `meta`

## Loading

```python
from datasets import load_dataset
import json

ds = load_dataset("{repo_id or 'NagaYu/sumi-ja-pii-corpus'}")
row = ds["test"][0]
spans = json.loads(row["spans"])
for s in spans:
    assert row["text"][s["start"]:s["end"]] == s["text"]
```

## Reproduce

```bash
python3 scripts/build_dataset.py --train {len(sets.get('train', []))} \\
  --val {len(sets.get('validation', []))} --test {len(sets.get('test', []))} \\
  --neg {len(sets.get('negatives', []))} --seed 0
```
"""


def main() -> None:
    """CLI エントリポイント。

    Claim: 検出率 / 低誤検出 — データ生成を1コマンドで再現可能にする。
    """
    ap = argparse.ArgumentParser(description="Sumi 合成PIIコーパスを生成する")
    ap.add_argument("--train", type=int, default=12000)
    ap.add_argument("--val", type=int, default=1500)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--neg", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=400, help="土台テキストのチャンク数")
    ap.add_argument("--inject-ratio", type=float, default=0.45)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--weights", default="", help="否定例の型別重み JSON (閉ループの出力)")
    ap.add_argument("--push", default="", help="HF Dataset repo id (明示的に指定した場合のみ push)")
    args = ap.parse_args()

    weights = None
    if args.weights and os.path.exists(args.weights):
        with open(args.weights, encoding="utf-8") as f:
            weights = json.load(f)
        print(f"閉ループの重みを適用: {args.weights}")

    print("=" * 74)
    print("Sumi 合成PIIコーパス生成")
    print("=" * 74)
    sets = build(args.train, args.val, args.test, args.neg,
                 seed=args.seed, n_base=args.base,
                 inject_ratio=args.inject_ratio, weights=weights)

    for name, docs in sets.items():
        p = os.path.join(args.out, f"{name}.jsonl")
        write_jsonl(docs, p)
        print(f"  -> {p}  ({os.path.getsize(p)/1e6:.1f} MB)")

    card = dataset_card(sets, args.push)
    cp = os.path.join(args.out, "dataset_card.md")
    with open(cp, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"  -> {cp}")

    if args.push:
        push(sets, card, args.push)
    else:
        print("\n(--push <repo_id> を指定すると Hugging Face へ公開します。"
              "公開は明示的な操作としてあり、既定では行いません)")
    print("=" * 74)


def push(sets: dict[str, list[Document]], card: str, repo_id: str) -> None:
    """HF Dataset として push する (明示指定時のみ)。

    Claim: 検出率 — 評価に使ったのと同一のデータを公開し、第三者が
    ベンチマークを再現できるようにする。
    """
    from datasets import Dataset, DatasetDict

    dd = DatasetDict(
        {
            name: Dataset.from_list([
                {**d.to_dict(),
                 "spans": json.dumps([s.to_dict() for s in d.sorted_spans()], ensure_ascii=False),
                 "meta": json.dumps({k: v for k, v in (d.meta or {}).items()
                                     if not k.startswith("_")}, ensure_ascii=False)}
                for d in docs
            ])
            for name, docs in sets.items()
        }
    )
    print(f"\npush -> https://huggingface.co/datasets/{repo_id}")
    dd.push_to_hub(repo_id)
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
    )
    print("push 完了")


if __name__ == "__main__":
    main()
