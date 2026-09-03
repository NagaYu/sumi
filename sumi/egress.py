"""外部送信の境界 — 対応表が machine の外へ出ないことを実行時に強制する層。

Claim: 可逆性 — 「マスクしてから送る」だけでは、実装ミスで元値が漏れうる。
本モジュールは送信直前に元値の混入を検査する番人 (:class:`EgressGuard`) と、
テストが送信内容を全件検査できる記録用トランスポートを提供する。
pytest はこの記録を用いて「対応表が外部送信経路に含まれないこと」を検証する。
"""

from __future__ import annotations

import unicodedata
from typing import Callable, Iterable, Protocol, runtime_checkable


class EgressViolation(RuntimeError):
    """元値が外部送信ペイロードに混入したときに送出される例外。

    Claim: 可逆性 — 漏洩を「起きたら気づく」ではなく「起きたら止まる」にする。
    """


@runtime_checkable
class Transport(Protocol):
    """外部へ1回の送信を行う最小インターフェース。

    Claim: 可逆性 — 送信経路を1点に絞ることで、そこだけを検査すれば
    「対応表が外に出ない」ことを保証できる設計にする。
    """

    def send(self, payload: str) -> str:
        """1件のペイロードを外部へ送り、応答を返す。

        Claim: 可逆性 — この1メソッドが外部との唯一の接点であり、
        ここだけを検査すれば対応表の非送信を保証できる。
        """
        ...


class RecordingTransport:
    """送信されたペイロードを全件記録するトランスポート (テスト用の計測器)。

    Claim: 可逆性 — pytest が ``sent`` を全走査して、対応表の元値が
    1バイトも外部経路に現れないことを機械的に検証できるようにする。
    """

    def __init__(self, responder: Callable[[str], str] | None = None) -> None:
        self.sent: list[str] = []
        self._responder = responder

    def send(self, payload: str) -> str:
        """ペイロードを記録し、応答を返す。

        Claim: 可逆性 — 送信内容を漏れなく残すことが検証の前提になる。
        """
        self.sent.append(payload)
        if self._responder is not None:
            return self._responder(payload)
        return payload


class EchoTransport:
    """ペイロードをそのまま返すトランスポート。

    Claim: 可逆性 — 往復でテキストが変化しない条件下で、
    マスク→復元が原文を完全に再現することを確認するために使う。
    """

    def send(self, payload: str) -> str:
        """ペイロードをそのまま返す。

        Claim: 可逆性 — 往復の同一性検査の基準となる素通し経路。
        """
        return payload


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


class EgressGuard:
    """送信直前に、禁止値 (対応表の元値) の混入を検査する番人。

    Claim: 可逆性 — マスク漏れ・二重送信・整形時の復元ミスといった実装事故を、
    送信前に確実に止める。

    検査対象は3形態に限定する (過剰なあいまい照合はしない):
        1. 生の文字列そのもの
        2. NFKC 正規化した形 (全角/半角の揺れ)
        3. 数字のみを抜き出した形 (``090-1234-5678`` → ``09012345678``)
    2文字以下の短い値は誤検知が実害を上回るため対象外とする。
    """

    #: これ以下の長さの値は照合対象にしない (偶然一致が多すぎるため)
    MIN_LEN = 3

    def __init__(self, forbidden: Iterable[str]) -> None:
        self.forbidden: list[str] = [f for f in dict.fromkeys(forbidden) if f]
        self._raw = [f for f in self.forbidden if len(f) >= self.MIN_LEN]
        self._norm = [
            unicodedata.normalize("NFKC", f) for f in self._raw
        ]
        self._digit = [
            _digits(f) for f in self._raw
            if len(_digits(f)) >= 7 and len(_digits(f)) >= len(f) * 0.6
        ]

    def check(self, payload: str) -> None:
        """ペイロードに禁止値が含まれていれば :class:`EgressViolation` を送出する。

        Claim: 可逆性 — 「対応表が外部へ出ない」という主張を、
        テストだけでなく実行時にも成立させる。
        """
        if not payload:
            return
        norm = unicodedata.normalize("NFKC", payload)
        dig = _digits(payload)
        for original, n in zip(self._raw, self._norm):
            if original in payload:
                raise EgressViolation(
                    f"元値が外部送信ペイロードに含まれています: {_preview(original)!r}"
                )
            if n and n in norm:
                raise EgressViolation(
                    f"元値(正規化形)が外部送信ペイロードに含まれています: {_preview(original)!r}"
                )
        for d in self._digit:
            if d and d in dig:
                raise EgressViolation(
                    f"元値(数字のみの形)が外部送信ペイロードに含まれています: {_preview(d)!r}"
                )

    def is_clean(self, payload: str) -> bool:
        """例外を送出せずに検査結果を返す。

        Claim: 可逆性 — UI やログで「安全に送れる状態か」を表示するための非破壊版。
        """
        try:
            self.check(payload)
            return True
        except EgressViolation:
            return False


def _preview(value: str) -> str:
    """例外メッセージに元値を丸ごと書かないための伏字化。

    Claim: 可逆性 — 漏洩を防ぐための例外メッセージ自体が
    元値をログへ漏らしてしまう事故を防ぐ。
    """
    if len(value) <= 2:
        return "●" * len(value)
    return value[0] + "●" * (len(value) - 2) + value[-1]


class GuardedTransport:
    """検査を通してから委譲するトランスポートのラッパ。

    Claim: 可逆性 — 呼び出し側が検査を忘れても、経路そのものが検査を強制する。
    """

    def __init__(self, transport: Transport, guard: EgressGuard) -> None:
        self.transport = transport
        self.guard = guard
        self.blocked: int = 0

    def send(self, payload: str) -> str:
        """検査後に委譲する。違反があれば送信しない。

        Claim: 可逆性 — 違反時に ``transport.send`` を **呼ばない** ことが要点。
        """
        try:
            self.guard.check(payload)
        except EgressViolation:
            self.blocked += 1
            raise
        return self.transport.send(payload)


def guarded(transport: Transport, guard: EgressGuard) -> GuardedTransport:
    """トランスポートを番人で包む。

    Claim: 可逆性 — 「送信経路は必ず番人を通る」を型で表現する。
    """
    return GuardedTransport(transport, guard)


def _selftest() -> None:
    """自己テスト。

    Claim: 可逆性 — 番人が本当に止めること、素通し経路が記録を残すことを確認する。
    """
    print("=" * 70)
    print("sumi.egress 自己テスト")
    print("=" * 70)

    g = EgressGuard(["田中太郎", "090-1234-5678", "tanaka@example.com", "あ"])
    print(f"禁止値 {len(g.forbidden)} 件 (うち照合対象 {len(g._raw)} 件、"
          f"数字形 {len(g._digit)} 件、2文字以下は除外)")

    clean = "<NAME_1> 様の連絡先は <PHONE_1> です。"
    g.check(clean)
    print(f"  ✓ マスク済みペイロードは通過: {clean!r}")

    for bad, why in [
        ("田中太郎 様", "生の文字列"),
        ("連絡先は 090-1234-5678", "生の数字列"),
        ("連絡先は ０９０－１２３４－５６７８", "全角 (NFKC 正規化形)"),
        ("tel:09012345678", "区切りを外した数字のみの形"),
        ("mail: tanaka@example.com", "メールアドレス"),
    ]:
        try:
            g.check(bad)
            print(f"  ✗ 検出できず ({why}): {bad!r}")
            raise AssertionError(f"guard missed: {why}")
        except EgressViolation as e:
            print(f"  ✓ 阻止 ({why}): {str(e)[:60]}")

    # 例外メッセージに元値がそのまま出ていないこと
    try:
        g.check("田中太郎")
    except EgressViolation as e:
        assert "田中太郎" not in str(e), "例外メッセージが元値を丸ごと漏らしている"
        print(f"  ✓ 例外メッセージは伏字: {str(e)[-24:]}")

    # GuardedTransport は違反時に send を呼ばない
    rec = RecordingTransport()
    gt = guarded(rec, g)
    gt.send(clean)
    try:
        gt.send("田中太郎の件")
    except EgressViolation:
        pass
    assert len(rec.sent) == 1, "違反ペイロードが下位トランスポートに渡ってしまった"
    assert gt.blocked == 1
    print(f"  ✓ GuardedTransport: 送信 {len(rec.sent)} 件 / 阻止 {gt.blocked} 件 "
          f"(違反分は下位に渡っていない)")

    print("=" * 70)
    print("すべての自己テストに合格 (番人 / 記録 / 阻止)")
    print("=" * 70)


if __name__ == "__main__":
    _selftest()
