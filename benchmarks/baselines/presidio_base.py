"""(A) Presidio 既定設定 — 英語向けの既定構成をそのまま日本語に当てた条件。

Claim: 検出率 — 「英語中心の既存ツールが日本語で落ちる」という出発点の主張を、
弁解の余地なく既定構成のまま測って示す。設定を悪くする細工は一切していない。
"""

from __future__ import annotations

from sumi.types import PIIType, Source, Span
from benchmarks.baselines import BaselineInfo

# Presidio のエンティティ名 -> Sumi の種別
PRESIDIO_TO_SUMI: dict[str, PIIType] = {
    "PERSON": PIIType.NAME,
    "LOCATION": PIIType.ADDRESS,
    "GPE": PIIType.ADDRESS,
    "NRP": PIIType.NAME,
    "PHONE_NUMBER": PIIType.PHONE,
    "EMAIL_ADDRESS": PIIType.EMAIL,
    "DATE_TIME": PIIType.DOB,
    "CREDIT_CARD": PIIType.CREDIT_CARD,
    "IBAN_CODE": PIIType.BANK_ACCOUNT,
    "US_BANK_NUMBER": PIIType.BANK_ACCOUNT,
    "US_SSN": PIIType.MYNUMBER,
}


class PresidioBaseline:
    """英語既定構成の Presidio AnalyzerEngine。

    Claim: 検出率 — 比較の基準線。日本語の氏名・住所をどれだけ取りこぼすかを示す。
    """

    name = "presidio_default"
    label = "(A) Presidio 既定"

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._engine = None

    def info(self) -> BaselineInfo:
        """条件のメタ情報。

        Claim: CPU速度 — 散布図の (サイズ, 精度) 座標に使う。
        en_core_web_sm は約 13M パラメータ相当の小型パイプライン。
        """
        return BaselineInfo(
            name=self.name,
            label=self.label,
            params=13e6,
            runtime="spaCy (en_core_web_sm) + regex",
            notes="英語既定構成をそのまま日本語入力に適用",
        )

    def available(self) -> bool:
        """依存が揃っているか。

        Claim: 検出率 — 依存欠如による欠測と、真の検出漏れを取り違えないため。
        """
        try:
            import presidio_analyzer  # noqa: F401
            import spacy

            spacy.load("en_core_web_sm")
            return True
        except Exception:
            return False

    def warmup(self) -> None:
        """エンジンを構築して初回ロード時間を計測から除く。

        Claim: CPU速度 — モデルロードは1回きりのコストなので推論速度と分けて測る。
        """
        if self._engine is not None:
            return
        from presidio_analyzer import AnalyzerEngine

        self._engine = AnalyzerEngine()
        self._engine.analyze(text="warmup John Smith", language=self.language)

    def detect(self, text: str) -> list[Span]:
        """既定 Presidio で検出し、Sumi の Span に写像する。

        Claim: 検出率 / 低誤検出 — Presidio の出力を Sumi と同じ座標系・
        同じ種別体系に正規化し、同一指標で採点できるようにする。
        """
        self.warmup()
        assert self._engine is not None
        results = self._engine.analyze(text=text, language=self.language)
        spans: list[Span] = []
        for r in results:
            t = PRESIDIO_TO_SUMI.get(r.entity_type)
            if t is None:
                continue
            if r.end <= r.start:
                continue
            spans.append(
                Span(
                    start=r.start,
                    end=r.end,
                    label=t,
                    text=text[r.start : r.end],
                    score=float(r.score),
                    source=Source.BASELINE,
                    meta={"entity_type": r.entity_type, "baseline": self.name},
                )
            )
        return _dedupe(spans)


def _dedupe(spans: list[Span]) -> list[Span]:
    """同一区間の重複を score 最大で1件に畳む。

    Claim: 低誤検出 — 同じ箇所に複数のレコグナイザが当たったとき、
    重複を誤検出数として二重計上しないようにする (基準線に有利側の処理)。
    """
    best: dict[tuple[int, int], Span] = {}
    for s in spans:
        k = (s.start, s.end)
        if k not in best or s.score > best[k].score:
            best[k] = s
    return sorted(best.values(), key=lambda s: (s.start, s.end))


if __name__ == "__main__":
    b = PresidioBaseline()
    print("available:", b.available())
    if b.available():
        t = "田中太郎様よりご連絡をいただきました。連絡先は090-1234-5678、tanaka.taro@example.co.jp です。住所は東京都新宿区西新宿2-8-1、生年月日は1985年3月4日。"
        for s in b.detect(t):
            print(f"  {s.label.ja:8s} {s.slice_of(t)!r} score={s.score:.2f}")
