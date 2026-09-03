"""合成データの不変条件を検証する — 実在の個人情報を使わないという約束の実装。

Claim: 検出率 / 低誤検出 — 正解が構成的に正しいこと、
そして「実在の個人情報を一切使わない」「チェックディジットは形式のみ正しく値は無効」
という制約が守られていることを機械的に固定する。
"""

from __future__ import annotations

import re

import pytest

from sumi.rules import luhn_ok, mynumber_check_ok
from sumi.synth import GENRES, PIIFactory, build_documents, render_document
from sumi.types import ALL_TYPES, PIIType, normalize

#: 予約ドメインの構造的定義。
#: RFC 2606 (example.com/net/org, .test/.invalid/.localhost/.example) と
#: JPRS が文書用に予約している example.jp / example.<属性>.jp を許可する。
#: サブドメイン (mail.example.com 等) も予約ドメイン配下なので許可。
RESERVED_DOMAIN_RE = re.compile(
    r"^(?:[a-z0-9-]+\.)*example\."
    r"(?:com|net|org|jp|co\.jp|ne\.jp|or\.jp|ac\.jp|ad\.jp|ed\.jp|go\.jp|gr\.jp|lg\.jp"
    r"|test|invalid|localhost|example)$"
)
#: 予約 TLD 直下も許可
RESERVED_TLD_RE = re.compile(r"\.(?:test|invalid|localhost|example)$")


def _is_reserved_domain(domain: str) -> bool:
    """ドメインが文書用予約ドメインかどうか。

    Claim: 低誤検出 — 実在しうるメールアドレスを1件も生成しないことの判定基準。
    """
    d = domain.strip().lower().rstrip(".")
    return bool(RESERVED_DOMAIN_RE.match(d) or RESERVED_TLD_RE.search(d))


@pytest.fixture(scope="module")
def docs():
    """合成文書。

    Claim: 検出率 — 十分な件数で不変条件を確認する。
    """
    return build_documents(500, seed=2024)


def test_all_documents_validate(docs):
    """全文書で text[start:end] == span.text が成り立つこと。

    Claim: 検出率 — 正解が壊れていれば検出率の数値自体が無意味になる。
    """
    for d in docs:
        d.validate()


def test_gold_spans_do_not_overlap(docs):
    """正解スパンが互いに重ならないこと。

    Claim: 可逆性 — 重なる正解はマスクの復元を一意でなくする。
    """
    for d in docs:
        ss = d.sorted_spans()
        for a, b in zip(ss, ss[1:]):
            assert a.end <= b.start, f"{d.doc_id}: {a.key()} と {b.key()} が重なる"


def test_offsets_are_on_normalized_text(docs):
    """本文が NFKC 正規化済みであること (再正規化で変化しないこと)。

    Claim: 検出率 — 正規化のタイミングがずれるとオフセットが壊れる。
    """
    for d in docs:
        assert normalize(d.text) == d.text, f"{d.doc_id}: 本文が正規化されていない"


def test_all_types_are_generated(docs):
    """全10種別の正解スパンが生成されること。

    Claim: 検出率 — 種別ごとの検出率を測るには各種別に十分な件数が要る。
    """
    counts = {t: 0 for t in ALL_TYPES}
    for d in docs:
        for s in d.spans:
            counts[s.label] += 1
    missing = [t.value for t, c in counts.items() if c == 0]
    assert not missing, f"生成されていない種別: {missing}"
    for t, c in counts.items():
        assert c >= 10, f"{t.ja} の件数が少なすぎる: {c}"


def test_all_genres_are_generated(docs):
    """4ジャンルすべての文書が生成されること。

    Claim: 検出率 — 業務文書らしい体裁 (メール/議事録/申込書/問い合わせ) を網羅する。
    """
    seen = {d.genre for d in docs}
    assert set(GENRES) <= seen, f"欠けているジャンル: {set(GENRES) - seen}"


# ------------------------------------------- 実在の個人情報を使わない保証


def test_emails_use_reserved_domains_only(docs):
    """合成メールアドレスが予約ドメインのみを使うこと。

    Claim: 低誤検出 — 実在しうるメールアドレスを生成しないための最重要の防波堤。
    """
    for d in docs:
        for s in d.spans:
            if s.label is not PIIType.EMAIL:
                continue
            domain = s.text.rsplit("@", 1)[-1]
            assert _is_reserved_domain(domain), (
                f"予約外ドメインが生成された: {s.text} (domain={domain})"
            )


def test_credit_cards_are_format_valid_but_value_invalid():
    """カード番号様式が、書式は正しく Luhn は必ず不一致であること。

    Claim: 低誤検出 — 「形式は正しく値は無効」という契約の実装確認。
    有効なカード番号を1件でも生成してはならない。
    """
    f = PIIFactory(seed=11)
    for _ in range(400):
        v = f.credit_card()
        digits = re.sub(r"\D", "", v.text)
        assert len(digits) in (14, 15, 16), f"桁数が不正: {v.text}"
        assert not luhn_ok(digits), f"有効な Luhn を生成してしまった: {v.text}"
        assert v.meta.get("checksum_valid") is False


def test_mynumbers_are_format_valid_but_value_invalid():
    """マイナンバー様式が、書式は正しく検査数字は必ず不一致であること。

    Claim: 低誤検出 — 実在しうる有効な個人番号を生成しないための保証。
    """
    f = PIIFactory(seed=13)
    for _ in range(400):
        v = f.mynumber()
        digits = re.sub(r"\D", "", v.text)
        assert len(digits) == 12, f"桁数が不正: {v.text}"
        assert not mynumber_check_ok(digits), f"有効な個人番号を生成してしまった: {v.text}"
        assert v.meta.get("checksum_valid") is False


def test_phones_follow_jp_numbering_plan():
    """生成した電話番号が日本の番号計画に適合すること。

    Claim: 検出率 — 規則層が拾えるべき正例が、そもそも妥当であることを保証する。
    """
    from sumi.rules import is_valid_jp_phone

    f = PIIFactory(seed=17)
    for _ in range(400):
        v = f.phone()
        assert is_valid_jp_phone(normalize(v.text)), f"番号計画に反する: {v.text}"


def test_addresses_have_randomized_banchi():
    """住所の丁目・番・号が乱数化されていること (実在の番地を固定しない)。

    Claim: 低誤検出 — 公開の地理名は使うが、番地は必ず乱数にするという約束。
    """
    f = PIIFactory(seed=19)
    tails = set()
    for _ in range(300):
        v = f.address()
        m = re.search(r"[\d一二三四五六七八九十]+(?:-[\d]+)*(?:丁目)?", v.text)
        if m:
            tails.add(m.group(0))
    assert len(tails) >= 50, f"番地のばらつきが不足: {len(tails)} 種"


def test_dob_wareki_conversion_is_correct():
    """和暦の生年月日が正しく換算されていること。

    Claim: 検出率 — 和暦を扱えることは日本語PII検出の要件。
    """
    f = PIIFactory(seed=23)
    eras = {"昭和": 1925, "平成": 1988, "令和": 2018}
    checked = 0
    for _ in range(500):
        v = f.dob(era="wareki")
        m = re.match(r"(昭和|平成|令和)(\d+)年(\d+)月(\d+)日", v.text)
        if not m:
            continue
        era, y = m.group(1), int(m.group(2))
        west = v.meta.get("year")
        if west:
            assert eras[era] + y == west, f"和暦換算が誤り: {v.text} -> {west}"
            checked += 1
    assert checked >= 50, f"検証できた和暦が少ない: {checked}"


def test_generation_is_reproducible():
    """同じシードなら完全に同じデータが出ること。

    Claim: 検出率 — ベンチマークの再現性の前提。
    """
    a = build_documents(40, seed=555)
    b = build_documents(40, seed=555)
    assert [x.to_dict() for x in a] == [y.to_dict() for y in b]
    c = build_documents(40, seed=556)
    assert [x.text for x in a] != [y.text for y in c]


def test_ambiguous_surnames_are_present(docs):
    """普通名詞・地名と同形の姓が一定割合で登場すること。

    Claim: 低誤検出 — hard negative が効くのは、正例側にも同じ曖昧さが
    存在するからである。片方だけでは学習も評価も歪む。
    """
    ambiguous = {"森", "林", "泉", "大和", "青木", "石田", "本田", "東", "西", "南", "北",
                 "長野", "福島", "千葉", "山口", "宮崎", "石川", "岡山"}
    n = 0
    total = 0
    for d in docs:
        for s in d.spans:
            if s.label is PIIType.NAME:
                total += 1
                if any(s.text.startswith(a) for a in ambiguous):
                    n += 1
    assert total > 0
    ratio = n / total
    assert ratio >= 0.03, f"曖昧な姓の割合が低すぎる: {ratio:.3f}"


def test_email_factory_only_emits_reserved_domains():
    """PIIFactory が生成する全ドメインが予約ドメインであること。

    Claim: 低誤検出 — 文書経由ではなく生成器を直接叩いて網羅的に確認する。
    実在しうるメールアドレスの生成は1件も許されない。
    """
    f = PIIFactory(seed=31)
    domains = set()
    for _ in range(4000):
        domains.add(normalize(f.email().text).rsplit("@", 1)[-1])
    assert domains, "ドメインが集まらなかった"
    bad = sorted(d for d in domains if not _is_reserved_domain(d))
    assert not bad, f"予約外ドメイン: {bad}"
    assert len(domains) >= 5, f"ドメインの多様性が不足: {sorted(domains)}"


# ------------------------------------------------- テンプレート非依存の評価集合


def test_ood_documents_are_valid_and_template_free():
    """OOD 評価集合が構成的に正しく、テンプレート由来でないこと。

    Claim: 検出率 — 「テンプレートの穴埋めを覚えただけ」で高い数値が出る事態を
    避けるための評価集合が、正しく作られていることを確認する。
    """
    from scripts.build_dataset import build_ood_documents
    from sumi.corpus import load_base_corpus
    from sumi.synth import TEMPLATES

    base = load_base_corpus(60, seed=0)
    if not base:
        pytest.skip("土台コーパスが無い (オフライン)")
    docs = build_ood_documents(40, seed=5, base_items=base)
    assert docs, "OOD 文書が生成されなかった"

    for d in docs:
        d.validate()
        assert d.genre == "ood_prose"
        assert d.spans, f"{d.doc_id}: 正解スパンが無い"
        assert normalize(d.text) == d.text

    # 業務文書テンプレートの特徴的な骨格が現れないこと
    skeletons = ["拝啓 時下ますます", "■決定事項", "1. 申込者情報", "■対応内容"]
    for d in docs:
        for sk in skeletons:
            assert sk not in d.text, f"{d.doc_id}: テンプレート断片 {sk!r} が混入"

    # 出典ライセンスが記録されていること
    assert all(d.source_license for d in docs)
    licenses = {d.source_license for d in docs}
    assert licenses, "ライセンスが記録されていない"


def test_ood_covers_multiple_types():
    """OOD 集合が複数の種別を含むこと。

    Claim: 検出率 — 汎化評価が特定種別に偏らないようにする。
    """
    from scripts.build_dataset import build_ood_documents
    from sumi.corpus import load_base_corpus

    base = load_base_corpus(60, seed=0)
    if not base:
        pytest.skip("土台コーパスが無い (オフライン)")
    docs = build_ood_documents(120, seed=9, base_items=base)
    seen = {s.label for d in docs for s in d.spans}
    assert len(seen) >= 8, f"種別が少なすぎる: {sorted(t.value for t in seen)}"
