"""ReversibleMasking — PIIを安定した置換子に置き換え、対応表をローカルに保つ。

Claim: 可逆性 — マスク済みテキストだけを外部 LLM へ送り、戻ってきた結果を
元の値に復元する経路を提供する。対応表はローカルファイル (0600) に留まり、
:mod:`sumi.egress` の番人により外部送信ペイロードへの混入が実行時に阻止される。

安定性の要件:
    同一文書内で **同じ元値は必ず同じ置換子** に写す。
    これにより、LLM 側で「<NAME_1> と <NAME_2> は別人」という関係が保たれ、
    要約・翻訳・書き換えといった下流タスクの品質を落とさない。
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sumi.egress import EgressGuard, Transport, guarded
from sumi.types import PIIType, Span, normalize

#: 置換子の書式 (style -> (接頭, 接尾))
_STYLES: dict[str, tuple[str, str]] = {
    "angle": ("<", ">"),
    "square": ("[", "]"),
    "brace": ("{{", "}}"),
}

#: 復元時に置換子を見つけるための正規表現
_PLACEHOLDER_RE = re.compile(
    r"(?:<([A-Z_]+)_(\d+)>|\[([A-Z_]+)_(\d+)\]|\{\{([A-Z_]+)_(\d+)\}\})"
)


@dataclass
class MaskEntry:
    """1件の置換対応。

    Claim: 可逆性 — 置換子・元値・種別・原文中の位置を保持することが、
    完全復元と監査 (何をどこで隠したか) の両方の根拠になる。
    """

    placeholder: str
    original: str
    label: PIIType
    start: int
    end: int

    def to_dict(self) -> dict:
        """JSON 化可能な辞書に変換する。

        Claim: 可逆性 — 対応表をローカルに永続化して後から復元できるようにする。
        """
        return {
            "placeholder": self.placeholder,
            "original": self.original,
            "label": self.label.value,
            "start": self.start,
            "end": self.end,
        }

    @staticmethod
    def from_dict(d: dict) -> "MaskEntry":
        """``to_dict`` の逆変換。

        Claim: 可逆性 — 保存した対応表を読み戻して復元できることを保証する。
        """
        return MaskEntry(
            placeholder=d["placeholder"],
            original=d["original"],
            label=PIIType(d["label"]),
            start=int(d["start"]),
            end=int(d["end"]),
        )


@dataclass
class MaskMap:
    """1文書分の対応表。**これがローカルから出てはならない。**

    Claim: 可逆性 — 復元に必要な情報の全体であり、同時に最も機微な資産。
    UI 表示には :meth:`redact_summary` を使い、元値を渡さない。
    """

    entries: list[MaskEntry] = field(default_factory=list)
    doc_id: str = ""
    version: str = "1"

    def originals(self) -> list[str]:
        """対応表に含まれる元値を列挙する (番人への入力)。

        Claim: 可逆性 — この一覧がそのまま「外部へ出てはいけない文字列」の定義になる。
        """
        return list(dict.fromkeys(e.original for e in self.entries))

    def placeholders(self) -> list[str]:
        """置換子を列挙する。

        Claim: 可逆性 — 復元漏れの検査に用いる。
        """
        return list(dict.fromkeys(e.placeholder for e in self.entries))

    def redact_summary(self) -> list[dict]:
        """元値を含まない要約を返す (画面表示・ログ用)。

        Claim: 可逆性 — 「何件を何の種別で隠したか」を見せつつ、
        元値そのものは決して渡さない、という分離を型で表す。
        """
        rows = []
        for e in self.entries:
            o = e.original
            preview = o[0] + "●" * max(0, len(o) - 1) if o else ""
            rows.append(
                {
                    "placeholder": e.placeholder,
                    "label": e.label.value,
                    "label_ja": e.label.ja,
                    "start": e.start,
                    "end": e.end,
                    "length": len(o),
                    "preview": preview,
                }
            )
        return rows

    def to_dict(self) -> dict:
        """JSON 化可能な辞書に変換する。

        Claim: 可逆性 — ローカル保存形式。
        """
        return {
            "version": self.version,
            "doc_id": self.doc_id,
            "entries": [e.to_dict() for e in self.entries],
        }

    @staticmethod
    def from_dict(d: dict) -> "MaskMap":
        """``to_dict`` の逆変換。

        Claim: 可逆性 — 保存済み対応表からの復元を保証する。
        """
        return MaskMap(
            entries=[MaskEntry.from_dict(x) for x in d.get("entries", [])],
            doc_id=d.get("doc_id", ""),
            version=str(d.get("version", "1")),
        )


class ReversibleMasker:
    """PII を安定した置換子へ置き換え、元へ戻す。

    Claim: 可逆性 — 墨消しは「消す」のではなく「戻せる形で隠す」。
    これにより外部 LLM を使いつつ、元の文書を失わない運用が成り立つ。
    """

    def __init__(self, *, style: str = "angle") -> None:
        if style not in _STYLES:
            raise ValueError(f"unknown style: {style!r} (known: {sorted(_STYLES)})")
        self.style = style

    def _fmt(self, label: PIIType, n: int) -> str:
        lo, hi = _STYLES[self.style]
        return f"{lo}{label.value}_{n}{hi}"

    def mask(
        self, text: str, spans: Sequence[Span], *, doc_id: str = ""
    ) -> tuple[str, MaskMap]:
        """スパンを置換子に置き換え、対応表とともに返す。

        Claim: 可逆性 — 置換は原文の右側から行い、まだ書き換えていない部分の
        オフセットを壊さない。同一の元値には同一の置換子を再利用するため、
        LLM 側で同一人物の同一性が保たれる。

        Returns:
            ``(masked_text, MaskMap)``。``MaskMap.entries`` の ``start``/``end`` は
            **原文** における位置。
        """
        # 重なりを除去して左から整列 (重なると復元が一意でなくなる)
        ordered: list[Span] = []
        for s in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
            if ordered and s.start < ordered[-1].end:
                continue
            ordered.append(s)

        counters: dict[PIIType, int] = {}
        assigned: dict[tuple[str, PIIType], str] = {}
        entries: list[MaskEntry] = []

        # 置換子の割り当ては「最初の出現順」で行う
        for s in ordered:
            original = s.text if s.text else s.slice_of(text)
            key = (unicodedata.normalize("NFKC", original), s.label)
            ph = assigned.get(key)
            if ph is None:
                counters[s.label] = counters.get(s.label, 0) + 1
                ph = self._fmt(s.label, counters[s.label])
                assigned[key] = ph
            entries.append(
                MaskEntry(placeholder=ph, original=original, label=s.label,
                          start=s.start, end=s.end)
            )

        # 実際の置換は右から (左側のオフセットを保つため)
        out = text
        for e in sorted(entries, key=lambda e: e.start, reverse=True):
            out = out[: e.start] + e.placeholder + out[e.end :]

        return out, MaskMap(entries=entries, doc_id=doc_id)

    def unmask(self, text: str, mmap: MaskMap) -> str:
        """置換子を元値へ戻す。

        Claim: 可逆性 — LLM が置換子を並べ替えても、重複させても、削っても壊れない。
        位置ではなく **置換子そのもの** を鍵にして戻すため、
        戻り値の構造が変わっていても復元できる。
        対応表に無い ``<FOO_9>`` のような未知の置換子はそのまま残す
        (勝手に消すと情報を失うため)。
        """
        table = {e.placeholder: e.original for e in mmap.entries}
        if not table:
            return text

        def repl(m: re.Match[str]) -> str:
            ph = m.group(0)
            return table.get(ph, ph)

        # 長い置換子から順に置換 (<NAME_1> と <NAME_11> の取り違えを防ぐ)
        return _PLACEHOLDER_RE.sub(repl, text)

    def save_map(self, mmap: MaskMap, path: str) -> None:
        """対応表をローカルに保存する (パーミッション 0600)。

        Claim: 可逆性 — 対応表は復元の鍵であり最も機微。
        所有者以外が読めないモードで書き出すことを実装として保証する。
        """
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        # 先に 0600 で作成してから書く (書いた後に chmod すると一瞬 0644 で存在する)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(mmap.to_dict(), f, ensure_ascii=False, indent=2)
        os.chmod(path, 0o600)

    @staticmethod
    def load_map(path: str) -> MaskMap:
        """保存した対応表を読み込む。

        Claim: 可逆性 — 別プロセス・別時刻からでも復元できることを保証する。
        """
        with open(path, encoding="utf-8") as f:
            return MaskMap.from_dict(json.load(f))


class LLMRoundTrip:
    """マスク → 外部LLM送信 → 復元 の経路。

    Claim: 可逆性 — 外部 LLM を使う運用そのものを安全側に閉じる。
    送信は必ず :class:`~sumi.egress.EgressGuard` を通り、
    対応表の元値が1つでも混じっていれば送信されずに例外となる。
    """

    def __init__(self, transport: Transport, masker: ReversibleMasker | None = None) -> None:
        self.transport = transport
        self.masker = masker or ReversibleMasker()

    def run(self, text: str, spans: Sequence[Span], *, instruction: str = "") -> dict:
        """マスクして送信し、応答を復元して返す。

        Claim: 可逆性 — 番人は省略可能な引数ではなく経路に埋め込まれている。
        これが「対応表が外部へ出ない」という主張の実装上の根拠。

        Returns:
            ``{"masked", "response_masked", "response", "map", "payload"}``
        """
        masked, mmap = self.masker.mask(text, spans)
        payload = f"{instruction}\n\n{masked}" if instruction else masked

        guard = EgressGuard(mmap.originals())
        safe = guarded(self.transport, guard)   # 検査を経路に強制する
        response_masked = safe.send(payload)

        return {
            "masked": masked,
            "payload": payload,
            "response_masked": response_masked,
            "response": self.masker.unmask(response_masked, mmap),
            "map": mmap,
        }


def _selftest() -> None:
    """自己テスト。

    Claim: 可逆性 — 完全復元・置換子の安定性・対応表の非送信・0600 を確認する。
    """
    import tempfile
    from sumi.egress import EchoTransport, EgressViolation, RecordingTransport

    print("=" * 74)
    print("sumi.mask 自己テスト")
    print("=" * 74)

    text = normalize(
        "田中太郎様\n"
        "平素よりお世話になっております。田中太郎様のご連絡先 090-1234-5678 と、\n"
        "予備の連絡先 090-1234-5678 を確認いたしました。\n"
        "メールは taro@example.co.jp、ご住所は東京都新宿区西新宿2-8-1 です。\n"
        "担当は佐藤花子が務めます。"
    )
    def sp(sub: str, label: PIIType, occ: int = 0) -> Span:
        idx, pos = -1, 0
        for _ in range(occ + 1):
            idx = text.index(sub, pos)
            pos = idx + 1
        return Span(idx, idx + len(sub), label, sub, 0.99)

    spans = [
        sp("田中太郎", PIIType.NAME, 0), sp("田中太郎", PIIType.NAME, 1),
        sp("090-1234-5678", PIIType.PHONE, 0), sp("090-1234-5678", PIIType.PHONE, 1),
        sp("taro@example.co.jp", PIIType.EMAIL),
        sp("東京都新宿区西新宿2-8-1", PIIType.ADDRESS),
        sp("佐藤花子", PIIType.NAME),
    ]

    m = ReversibleMasker()
    masked, mmap = m.mask(text, spans, doc_id="demo-1")
    print("\n[マスク結果]")
    for line in masked.splitlines():
        print("   ", line)

    print("\n[対応表の要約 — 元値を含まない]")
    for row in mmap.redact_summary():
        print(f"    {row['placeholder']:12s} {row['label_ja']:8s} "
              f"len={row['length']:2d} preview={row['preview']}")
    joined = json.dumps(mmap.redact_summary(), ensure_ascii=False)
    for o in mmap.originals():
        assert o not in joined, f"redact_summary が元値を漏らしている: {o}"
    print("    ✓ 要約に元値は一切含まれない")

    # --- 安定性 ---
    print("\n[置換子の安定性]")
    ph = {}
    for e in mmap.entries:
        ph.setdefault(e.original, set()).add(e.placeholder)
    for orig, s in ph.items():
        assert len(s) == 1, f"同じ元値に複数の置換子: {orig} -> {s}"
    assert masked.count("<NAME_1>") == 2, "繰り返す氏名が同一置換子になっていない"
    assert masked.count("<PHONE_1>") == 2, "繰り返す電話番号が同一置換子になっていない"
    assert "<NAME_2>" in masked, "別人が別の置換子になっていない"
    print(f"    ✓ 同一値は同一置換子 (<NAME_1>×2, <PHONE_1>×2)、別人は <NAME_2>")

    # --- 完全復元 ---
    restored = m.unmask(masked, mmap)
    assert restored == text, "復元が原文と一致しない"
    print("\n[完全復元]  unmask(mask(t)) == t  ✓")

    # --- LLM が構造を変えても戻せること ---
    scrambled = "順序変更: <PHONE_1> / <NAME_2> / <NAME_1> / 未知 <FOO_9> / 重複 <NAME_1>"
    out = m.unmask(scrambled, mmap)
    assert "田中太郎" in out and "佐藤花子" in out and out.count("田中太郎") == 2
    assert "<FOO_9>" in out, "未知の置換子を勝手に消してはいけない"
    print(f"[頑健性]  並べ替え/重複/未知置換子 -> {out}")

    # --- 往復 (Echo) ---
    rt = LLMRoundTrip(EchoTransport(), m)
    r = rt.run(text, spans, instruction="次の文書を要約してください。")
    assert r["response"].endswith(text), "Echo 往復で原文が復元されない"
    print("\n[往復]  Echo トランスポートで原文復元 ✓")

    # --- 対応表が送信経路に出ないこと ---
    rec = RecordingTransport(responder=lambda p: "要約: <NAME_1> 様の件、連絡先 <PHONE_1>。")
    rt2 = LLMRoundTrip(rec, m)
    r2 = rt2.run(text, spans, instruction="要約してください。")
    print("\n[送信内容の検査]")
    print(f"    送信 {len(rec.sent)} 件")
    leaked = [o for o in mmap.originals() for p in rec.sent if o in p]
    assert not leaked, f"送信ペイロードに元値が混入: {leaked}"
    print(f"    ✓ 元値 {len(mmap.originals())} 件のいずれも送信ペイロードに現れない")
    print(f"    復元後の応答: {r2['response']}")
    assert "田中太郎" in r2["response"] and "090-1234-5678" in r2["response"]

    # --- 番人が未マスク送信を止めること ---
    class LeakyMasker(ReversibleMasker):
        def mask(self, text, spans, *, doc_id=""):
            _, mm = super().mask(text, spans, doc_id=doc_id)
            return text, mm          # わざとマスクせずに返す
    rec2 = RecordingTransport()
    try:
        LLMRoundTrip(rec2, LeakyMasker()).run(text, spans)
        raise AssertionError("未マスク送信が阻止されなかった")
    except EgressViolation as e:
        print(f"\n[番人]  未マスク送信を阻止 ✓ ({str(e)[:52]}...)")
    assert rec2.sent == [], "阻止したのに下位トランスポートへ渡っている"
    print("    ✓ 阻止時に送信は一切行われていない")

    # --- 保存とパーミッション ---
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sub", "map.json")
        m.save_map(mmap, p)
        mode = os.stat(p).st_mode & 0o777
        assert mode == 0o600, f"対応表のパーミッションが {oct(mode)} (0o600 であるべき)"
        again = ReversibleMasker.load_map(p)
        assert m.unmask(masked, again) == text, "保存→読込後に復元できない"
        print(f"\n[保存]  {oct(mode)} で保存、読み戻して完全復元 ✓")

    print("\n" + "=" * 74)
    print("すべての自己テストに合格 (完全復元 / 安定性 / 非送信 / 0600)")
    print("=" * 74)


if __name__ == "__main__":
    _selftest()
