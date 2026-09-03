"""規則層が形式確定の種別を取りこぼさないことを検証する。

Claim: 検出率 / 低誤検出 — 主張(3)。電話番号・メール・郵便番号・口座・
カード番号様式・マイナンバー様式・会員番号のように書式が決まっている種別は、
規則層が高い再現率で拾えなければならない。同時に、紛らわしい数字列で
誤検出しないことも確認する。
"""

from __future__ import annotations

import pytest

from sumi.rules import (
    RuleLayer,
    is_valid_jp_phone,
    luhn_ok,
    merge_spans,
    mynumber_check_ok,
)
from sumi.synth import PIIFactory, build_documents
from sumi.types import (
    MODEL_DRIVEN,
    RULE_DETERMINISTIC,
    PIIType,
    Source,
    Span,
    normalize,
)

#: 規則層が担当する種別のうち、合成データで十分な件数が出るもの
COVERED = [
    PIIType.EMAIL,
    PIIType.PHONE,
    PIIType.POSTAL_CODE,
    PIIType.BANK_ACCOUNT,
    PIIType.CREDIT_CARD,
    PIIType.MYNUMBER,
]

#: 種別ごとの最低再現率 (partial 一致)
MIN_RECALL = {
    PIIType.EMAIL: 0.99,
    PIIType.PHONE: 0.95,
    PIIType.POSTAL_CODE: 0.90,
    PIIType.BANK_ACCOUNT: 0.85,
    PIIType.CREDIT_CARD: 0.95,
    PIIType.MYNUMBER: 0.85,
}


@pytest.fixture(scope="module")
def layer() -> RuleLayer:
    """既定設定の規則層。

    Claim: 低減検出 — テストと配布物で同じ設定を使う。
    """
    return RuleLayer()


@pytest.fixture(scope="module")
def docs():
    """合成文書 (正解スパン付き)。

    Claim: 検出率 — 構成的に作った正解に対して再現率を測る。
    """
    return build_documents(400, seed=77)


# ------------------------------------------------------------- checksums


def test_luhn_known_values():
    """Luhn 実装が既知の値と一致すること。

    Claim: 低誤検出 — checksum は注記に使うため、正確でなければ意味がない。
    """
    assert luhn_ok("4111111111111111")
    assert luhn_ok("5500000000000004")
    assert luhn_ok("4111 1111 1111 1111")
    assert not luhn_ok("4111111111111112")
    assert not luhn_ok("1234567890123456")


def test_mynumber_check_digit_is_unique():
    """マイナンバーの検査数字が、各 11 桁に対し唯一であること。

    Claim: 低誤検出 — 総務省の算式を正しく実装していることの強い確認。
    """
    import random

    rng = random.Random(0)
    for _ in range(50):
        base = "".join(str(rng.randint(0, 9)) for _ in range(11))
        valid = [d for d in range(10) if mynumber_check_ok(base + str(d))]
        assert len(valid) == 1, f"{base}: 正解が {len(valid)} 個"


def test_jp_phone_validator():
    """日本の番号計画の判定が正しいこと。

    Claim: 低誤検出 — 桁数規則が電話と型番を分ける最大の武器。
    """
    for ok in ["03-1234-5678", "090-1234-5678", "0463-12-3456", "0120-123-456",
               "0800-123-4567", "050-1234-5678", "045-123-4567", "0312345678"]:
        assert is_valid_jp_phone(ok), ok
    for ng in ["03-1234", "1234-5678", "2024-01-15", "090-1234-567",
               "0120-8834-221", "12345678901", "0999-9999-9999"]:
        assert not is_valid_jp_phone(ng), ng


# -------------------------------------------------- 形式確定型の取りこぼし


@pytest.mark.parametrize("ptype", COVERED, ids=lambda t: t.value)
def test_rule_layer_recall_on_deterministic_types(layer, docs, ptype):
    """規則層が形式確定の種別を取りこぼさないこと。

    Claim: 検出率 — 主張(3)。合成文書中の当該種別の正解スパンに対し、
    規則層だけで所定の再現率を満たすことを要求する。
    """
    gold = 0
    hit = 0
    for d in docs:
        targets = [s for s in d.spans if s.label is ptype]
        if not targets:
            continue
        pred = [s for s in layer.detect(d.text) if s.label is ptype]
        gold += len(targets)
        for g in targets:
            if any(g.overlaps(p) for p in pred):
                hit += 1
    assert gold >= 20, f"{ptype.value}: 検証に足る正解数がない ({gold})"
    recall = hit / gold
    assert recall >= MIN_RECALL[ptype], (
        f"{ptype.ja}: 規則層の再現率 {recall:.3f} < 下限 {MIN_RECALL[ptype]} "
        f"({hit}/{gold})"
    )


def test_rule_layer_covers_all_deterministic_types(layer):
    """RULE_DETERMINISTIC の全種別に規則が存在すること。

    Claim: 検出率 — 契約で「規則が担当する」と宣言した種別に穴が無いことを確認する。
    """
    have = {s.label for s in layer.specs}
    missing = set(RULE_DETERMINISTIC) - have
    assert not missing, f"規則の無い形式確定型: {sorted(t.value for t in missing)}"


def test_rule_layer_does_not_claim_model_driven_types(layer):
    """規則層が文脈依存型 (氏名・住所・生年月日) を担当しないこと。

    Claim: 低誤検出 — 役割分担を守る。氏名を正規表現で拾おうとすると誤検出が爆発する。
    """
    assert not ({s.label for s in layer.specs} & set(MODEL_DRIVEN))


# --------------------------------------- checksum を検出条件にしない設計


def test_detection_is_not_gated_on_checksum(layer):
    """checksum が無効でも形式が合えば検出すること。

    Claim: 低誤検出 — 墨消しの目的は「それらしい物を残さないこと」。
    合成データは値が無効なので、ここを間違えると全件取りこぼす。
    """
    f = PIIFactory(seed=5)
    for _ in range(30):
        cc = f.credit_card()
        assert cc.meta.get("checksum_valid") is False
        t = normalize(f"お支払いはカード番号 {cc.text} でお願いします。")
        got = [s for s in layer.detect(t) if s.label is PIIType.CREDIT_CARD]
        assert got, f"Luhn 不一致のカード番号様式が検出されない: {cc.text}"
        assert got[0].meta["checksum_valid"] is False

        mn = f.mynumber()
        assert mn.meta.get("checksum_valid") is False
        t = normalize(f"マイナンバー {mn.text} を確認しました。")
        got = [s for s in layer.detect(t) if s.label is PIIType.MYNUMBER]
        assert got, f"検査数字不一致のマイナンバー様式が検出されない: {mn.text}"
        assert got[0].meta["checksum_valid"] is False


# ------------------------------------------------------------- 統合順序


def test_merge_prefers_rules_over_model():
    """規則スパンと重なるモデルスパンが破棄されること。

    Claim: 低誤検出 — 契約の優先順位ステップ1・2。
    """
    text = normalize("連絡先は 090-1234-5678 です。")
    rule = RuleLayer().detect(text)
    ph = next(s for s in rule if s.label is PIIType.PHONE)
    model = [Span(ph.start, ph.end - 3, PIIType.MEMBER_ID, text[ph.start:ph.end - 3],
                  0.99, Source.MODEL)]
    merged = merge_spans(model, rule, text)
    assert [s.label for s in merged] == [PIIType.PHONE]
    assert merged[0].meta["from"] == "rule"


def test_merge_keeps_higher_scoring_model_span():
    """モデル同士の重なりは高スコアが残ること。

    Claim: 低誤検出 — 契約の優先順位ステップ4。
    """
    text = normalize("田中太郎様。")
    model = [
        Span(0, 4, PIIType.NAME, "田中太郎", 0.9, Source.MODEL),
        Span(0, 2, PIIType.NAME, "田中", 0.4, Source.MODEL),
    ]
    merged = merge_spans(model, [], text)
    assert [s.text for s in merged] == ["田中太郎"]


def test_merge_output_is_sorted_and_non_overlapping(docs):
    """統合結果が常に start 昇順・非重複であること。

    Claim: 可逆性 — 重なりが残るとマスクの復元が一意でなくなる。
    """
    layer = RuleLayer()
    for d in docs[:60]:
        merged = merge_spans(d.spans, layer.detect(d.text), d.text)
        for a, b in zip(merged, merged[1:]):
            assert a.start <= b.start
            assert a.end <= b.start, f"{d.doc_id}: 重なりが残っている"
