"""Side-by-side comparison against a Presidio + GiNZA setup.

Claim: 検出率 / 低誤検出 — Sumi の主張は「日本語で既存構成より拾い、かつ誤検出しない」
ことなので、その比較対象を **ライブラリの一部として** 提供する。
Gradio デモとベンチマークの両方がここを唯一の実装として参照するため、
「デモで見せている比較」と「数字を出した比較」が乖離しない。

比較対象は藁人形ではない。GiNZA の拡張固有表現 (189ラベル) を Presidio の
エンティティへ丁寧に写像し、分割された地名断片を住所として連結する後処理まで
入れた、実務者が普通に組む最良に近い構成である。

presidio / spaCy / ja_ginza が入っていない環境でも import は失敗せず、
``PresidioGinzaComparator.available()`` が False を返すだけになる。
"""

from __future__ import annotations

import re
from typing import Sequence

from sumi.types import PIIType, Source, Span

#: GiNZA (Sekine 拡張固有表現) のラベル -> Sumi 種別。
#: 住所を構成しうる地名系ラベルを広く拾う (比較対象に有利側の写像)。
GINZA_TO_SUMI: dict[str, PIIType] = {
    "Person": PIIType.NAME,
    "Character": PIIType.NAME,
    "Address": PIIType.ADDRESS,
    "Postal_Address": PIIType.ADDRESS,
    "Province": PIIType.ADDRESS,
    "City": PIIType.ADDRESS,
    "County": PIIType.ADDRESS,
    "Country": PIIType.ADDRESS,
    "GPE_Other": PIIType.ADDRESS,
    "Region_Other": PIIType.ADDRESS,
    "Geological_Region_Other": PIIType.ADDRESS,
    "Continental_Region": PIIType.ADDRESS,
    "Domestic_Region": PIIType.ADDRESS,
    "Date": PIIType.DOB,
    "Era": PIIType.DOB,
    "Phone_Number": PIIType.PHONE,
    "Email": PIIType.EMAIL,
    "Postal_Code": PIIType.POSTAL_CODE,
    "ID_Number": PIIType.MEMBER_ID,
}

#: Presidio 自身のエンティティ名 -> Sumi 種別。
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

#: 隣接時に1つの住所へ連結してよい地名系ラベル。
ADDRESS_LABELS = frozenset({
    "Address", "Postal_Address", "Province", "City", "County",
    "Country", "GPE_Other", "Region_Other", "Domestic_Region",
})

_BANCHI_RE = re.compile(
    r"[\s　]*((?:\d+[-\d]*)|(?:[一二三四五六七八九十]+丁目[\d一二三四五六七八九十番地号\-]*))"
)


def join_adjacent_address(
    ents: Sequence[tuple[int, int, str]], text: str
) -> list[tuple[int, int, str]]:
    """隣接する地名系エンティティを1つの住所スパンへ連結する。

    Claim: 検出率 — GiNZA は「東京都新宿区西新宿」と「2-8-1」を別々に切りがちなので、
    間に区切り文字しか無い地名断片と、直後に続く丁目/番地表現を連結して、
    比較対象が住所を最大限拾えるよう **有利に** 補正する。
    """
    out: list[tuple[int, int, str]] = []
    ordered = sorted(ents, key=lambda e: e[0])
    i = 0
    while i < len(ordered):
        a, b, lab = ordered[i]
        if lab in ADDRESS_LABELS:
            j = i + 1
            while j < len(ordered):
                a2, b2, lab2 = ordered[j]
                gap = text[b:a2]
                if lab2 in ADDRESS_LABELS and len(gap) <= 1 and gap.strip("　 ") == "":
                    b = b2
                    j += 1
                else:
                    break
            m = _BANCHI_RE.match(text[b:])
            if m and m.group(1):
                b = b + m.end()
            out.append((a, b, "Address"))
            i = j
        else:
            out.append((a, b, lab))
            i += 1
    return out


def resolve_overlaps(spans: Sequence[Span]) -> list[Span]:
    """重なるスパンを score 優先で解消し、非重複・start 昇順にする。

    Claim: 低誤検出 — 重なりを残したまま採点すると比較対象の誤検出数が
    不当に膨らむため、重複は畳んでから比べる。
    """
    chosen: list[Span] = []
    for s in sorted(spans, key=lambda s: (-s.score, -(s.end - s.start), s.start)):
        if all(not s.overlaps(c) for c in chosen):
            chosen.append(s)
    return sorted(chosen, key=lambda s: (s.start, s.end))


class PresidioGinzaComparator:
    """Presidio の形式レコグナイザ + GiNZA の日本語NER を合成した比較対象。

    Claim: 検出率 / 低誤検出 — 「日本語NERを足せばどこまで届くのか」という
    実務上いちばん現実的な比較対象を、デモとベンチマークで共有する。
    """

    name = "presidio_ginza"
    label = "Presidio + GiNZA"

    def __init__(self, join_address: bool = True) -> None:
        self.join_address = join_address
        self._nlp = None
        self._engine = None
        self._unavailable_reason = ""

    def available(self) -> bool:
        """依存が揃っているか。

        Claim: 検出率 — 依存欠如による欠測を、真の検出漏れと取り違えないため。
        """
        try:
            import spacy
            from presidio_analyzer import AnalyzerEngine  # noqa: F401

            spacy.util.get_package_path("ja_ginza")
            return True
        except Exception as exc:
            self._unavailable_reason = f"{type(exc).__name__}: {exc}"
            return False

    @property
    def unavailable_reason(self) -> str:
        """利用できない理由 (UI 表示用)。

        Claim: 検出率 — 比較が出せないときに、その理由を隠さない。
        """
        return self._unavailable_reason

    def warmup(self) -> None:
        """GiNZA と Presidio を読み込む。

        Claim: CPU速度 — ロード時間を推論時間から分離する。
        """
        if self._nlp is not None:
            return
        import spacy
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        self._nlp = spacy.load("ja_ginza")
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "ja", "model_name": "ja_ginza"}],
            }
        )
        self._engine = AnalyzerEngine(
            nlp_engine=provider.create_engine(), supported_languages=["ja"]
        )
        self._nlp("ウォームアップ")
        self._engine.analyze(text="ウォームアップ", language="ja")

    def detect(self, text: str) -> list[Span]:
        """GiNZA の固有表現と Presidio の形式ルールを合成して返す。

        Claim: 検出率 / 低誤検出 — 氏名・住所は GiNZA、電話/メール等は Presidio、
        という実務でよく組まれる分担をそのまま再現する。
        """
        self.warmup()
        assert self._nlp is not None and self._engine is not None

        spans: list[Span] = []
        doc = self._nlp(text)
        ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
        if self.join_address:
            ents = join_adjacent_address(ents, text)
        for a, b, lab in ents:
            t = GINZA_TO_SUMI.get(lab)
            if t is None or b <= a:
                continue
            spans.append(Span(a, b, t, text[a:b], 0.85, Source.BASELINE,
                              meta={"entity_type": lab, "layer": "ginza"}))

        for r in self._engine.analyze(text=text, language="ja"):
            t = PRESIDIO_TO_SUMI.get(r.entity_type)
            if t is None or r.end <= r.start:
                continue
            spans.append(Span(r.start, r.end, t, text[r.start:r.end], float(r.score),
                              Source.BASELINE,
                              meta={"entity_type": r.entity_type, "layer": "presidio"}))

        return resolve_overlaps(spans)


def presidio_spans(text: str, _cache: dict = {}) -> list[Span]:
    """Presidio + GiNZA の検出結果を返す (使えなければ空リスト)。

    Claim: 検出率 — デモの比較表示から呼ぶための最小の入口。
    比較器はプロセス内で1つだけ作って使い回す。
    """
    comp = _cache.get("comp")
    if comp is None:
        comp = _cache["comp"] = PresidioGinzaComparator()
    if not comp.available():
        return []
    try:
        return comp.detect(text)
    except Exception:
        return []


if __name__ == "__main__":
    c = PresidioGinzaComparator()
    print("available:", c.available(), c.unavailable_reason)
    if c.available():
        for t in [
            "田中太郎様よりご連絡をいただきました。連絡先は090-1234-5678です。住所は東京都新宿区西新宿2-8-1。",
            "森の中を歩いた。長野県の気候は寒暖差が大きい。型番TX-2024-0355。",
        ]:
            print("\n" + t[:44])
            for s in c.detect(t):
                print(f"   {s.label.en:12s} {s.text!r} ({s.meta.get('entity_type')})")
