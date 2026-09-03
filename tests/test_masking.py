"""可逆マスキングと対応表の非送信を検証する。

Claim: 可逆性 — 本プロジェクトの安全性の主張は2つ。
  (1) 対応表が外部送信経路に含まれないこと
  (2) 可逆マスキングで元文が完全復元されること
本ファイルはその2つを機械的に検証する。
"""

from __future__ import annotations

import json
import os
import random

import pytest

from sumi.egress import (
    EchoTransport,
    EgressGuard,
    EgressViolation,
    RecordingTransport,
    guarded,
)
from sumi.mask import LLMRoundTrip, MaskMap, ReversibleMasker
from sumi.synth import build_documents
from sumi.types import Document, PIIType, Span, normalize


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    """正解スパン付きの合成文書。

    Claim: 可逆性 — 実際の PII 分布に近い文書で往復を検証する。
    """
    return build_documents(40, seed=1234)


# ---------------------------------------------------------------- 完全復元


def test_roundtrip_restores_original_exactly(docs):
    """可逆マスキングで元文が完全復元されること (全文書・全スパン)。

    Claim: 可逆性 — 主張(2)。1文字でも違えば失敗する厳密比較。
    """
    m = ReversibleMasker()
    for d in docs:
        masked, mmap = m.mask(d.text, d.spans, doc_id=d.doc_id)
        assert masked != d.text or not d.spans, f"{d.doc_id}: マスクされていない"
        restored = m.unmask(masked, mmap)
        assert restored == d.text, f"{d.doc_id}: 復元が原文と一致しない"


def test_roundtrip_via_echo_transport(docs):
    """LLM 往復 (Echo) を経ても原文が完全復元されること。

    Claim: 可逆性 — 主張(2)を、実際の送受信経路を通した形で確認する。
    """
    rt = LLMRoundTrip(EchoTransport())
    for d in docs[:12]:
        r = rt.run(d.text, d.spans)
        assert r["response"] == d.text, f"{d.doc_id}: 往復後に原文が復元されない"


def test_placeholders_are_stable_for_repeated_values():
    """同一の元値は同一の置換子に安定して割り当てられること。

    Claim: 可逆性 — 下流 LLM が同一人物の同一性を保てるようにするための要件。
    """
    text = normalize("田中太郎様と田中太郎様、および佐藤花子様。電話 090-1234-5678 と 090-1234-5678。")
    spans = []
    for sub, lab in [("田中太郎", PIIType.NAME), ("佐藤花子", PIIType.NAME),
                     ("090-1234-5678", PIIType.PHONE)]:
        start = 0
        while (i := text.find(sub, start)) >= 0:
            spans.append(Span(i, i + len(sub), lab, sub))
            start = i + 1
    masked, mmap = ReversibleMasker().mask(text, spans)
    assert masked.count("<NAME_1>") == 2
    assert masked.count("<PHONE_1>") == 2
    assert "<NAME_2>" in masked, "別人が同じ置換子に潰れている"
    by_original: dict[str, set[str]] = {}
    for e in mmap.entries:
        by_original.setdefault(e.original, set()).add(e.placeholder)
    assert all(len(v) == 1 for v in by_original.values())


def test_unmask_is_robust_to_llm_mangling(docs):
    """LLM が順序変更・重複・削除・未知置換子を返しても壊れないこと。

    Claim: 可逆性 — 位置ではなく置換子を鍵に復元しているため構造変化に耐える。
    """
    m = ReversibleMasker()
    d = next(x for x in docs if len(x.spans) >= 3)
    _, mmap = m.mask(d.text, d.spans)
    phs = mmap.placeholders()
    mangled = " / ".join(list(reversed(phs)) + [phs[0], "<UNKNOWN_9>"])
    out = m.unmask(mangled, mmap)
    assert "<UNKNOWN_9>" in out, "未知の置換子を勝手に消してはならない"
    for e in mmap.entries:
        assert e.original in out


def test_save_map_is_owner_only(tmp_path, docs):
    """対応表ファイルが 0600 で保存されること。

    Claim: 可逆性 — 復元の鍵を同一マシン上の他ユーザからも守る。
    """
    m = ReversibleMasker()
    _, mmap = m.mask(docs[0].text, docs[0].spans)
    p = tmp_path / "sub" / "map.json"
    m.save_map(mmap, str(p))
    assert (os.stat(p).st_mode & 0o777) == 0o600
    assert m.unmask(m.mask(docs[0].text, docs[0].spans)[0],
                    ReversibleMasker.load_map(str(p))) == docs[0].text


# --------------------------------------------------- 対応表が外部へ出ない


def test_mapping_never_appears_in_egress_payloads(docs):
    """対応表の元値が外部送信ペイロードに1件も現れないこと。

    Claim: 可逆性 — 主張(1)。送信された全バイトを走査して確認する。
    これが本プロジェクトの安全性の中核。
    """
    for d in docs[:20]:
        rec = RecordingTransport(responder=lambda p: p)
        rt = LLMRoundTrip(rec)
        r = rt.run(d.text, d.spans, instruction="次を要約してください。")
        originals = r["map"].originals()
        assert rec.sent, "送信が記録されていない (テストが無意味になる)"
        for payload in rec.sent:
            for o in originals:
                assert o not in payload, (
                    f"{d.doc_id}: 元値 {o!r} が送信ペイロードに含まれている"
                )


def test_mapping_not_in_any_serialized_form(docs):
    """対応表の元値が、送信ペイロードのどの表現形でも現れないこと。

    Claim: 可逆性 — 生・NFKC正規化形・数字のみ形の3形態で検査する。
    """
    for d in docs[:15]:
        rec = RecordingTransport()
        rt = LLMRoundTrip(rec)
        r = rt.run(d.text, d.spans)
        guard = EgressGuard(r["map"].originals())
        for payload in rec.sent:
            guard.check(payload)   # 違反があれば EgressViolation


def test_redact_summary_contains_no_originals(docs):
    """UI 表示用の要約に元値が含まれないこと。

    Claim: 可逆性 — 画面やログ経由の漏洩も防ぐ。
    """
    for d in docs[:20]:
        _, mmap = ReversibleMasker().mask(d.text, d.spans)
        blob = json.dumps(mmap.redact_summary(), ensure_ascii=False)
        for o in mmap.originals():
            if len(o) >= 3:
                assert o not in blob, f"要約に元値 {o!r} が漏れている"


def test_guard_blocks_unmasked_send_and_does_not_transmit(docs):
    """マスク漏れがあれば送信そのものが行われないこと。

    Claim: 可逆性 — 番人は「検知」ではなく「阻止」であることを確認する。
    """
    d = next(x for x in docs if x.spans)

    class LeakyMasker(ReversibleMasker):
        def mask(self, text, spans, *, doc_id=""):
            _, mm = super().mask(text, spans, doc_id=doc_id)
            return text, mm      # わざとマスクしない

    rec = RecordingTransport()
    with pytest.raises(EgressViolation):
        LLMRoundTrip(rec, LeakyMasker()).run(d.text, d.spans)
    assert rec.sent == [], "阻止したはずなのに送信されている"


def test_guard_error_message_does_not_leak_the_value():
    """番人の例外メッセージ自体が元値を漏らさないこと。

    Claim: 可逆性 — 例外がログに出ても値が残らないようにする。
    """
    secret = "田中太郎"
    guard = EgressGuard([secret, "090-1234-5678"])
    with pytest.raises(EgressViolation) as ei:
        guard.check(f"送信テキスト {secret} を含む")
    assert secret not in str(ei.value)


def test_guard_allows_properly_masked_payload(docs):
    """正しくマスクされたペイロードは通ること (過剰検知でないこと)。

    Claim: 可逆性 — 番人が厳しすぎて実運用を止めてしまわないことを確認する。
    """
    for d in docs[:20]:
        masked, mmap = ReversibleMasker().mask(d.text, d.spans)
        EgressGuard(mmap.originals()).check(masked)


def test_guarded_transport_wrapper_enforces_check():
    """guarded() で包んだ経路は検査を強制すること。

    Claim: 可逆性 — 呼び出し側が検査を忘れても安全側に倒れる。
    """
    rec = RecordingTransport()
    g = guarded(rec, EgressGuard(["秘密の値"]))
    g.send("安全なテキスト")
    with pytest.raises(EgressViolation):
        g.send("これは秘密の値を含む")
    assert rec.sent == ["安全なテキスト"]
