"""(B) Presidio + GiNZA — 日本語NERを足した「現行の実務」構成。

Claim: 検出率 / 低誤検出 — これは **わざと弱くした藁人形ではない**。
GiNZA の拡張固有表現 (189ラベル) を Presidio のエンティティへ丁寧に写像し、
隣接する地名断片を住所として連結する後処理まで入れた、
実務者が普通に組む最良に近い構成として実装している。
それでも日本語の紛らわしい否定例で何が起きるかを測るのが目的。
"""

from __future__ import annotations

from sumi.types import PIIType, Source, Span
from benchmarks.baselines import BaselineInfo

#: GiNZA (Sekine 拡張固有表現) のラベル -> Sumi 種別。
#: 住所を構成しうる地名系ラベルを広く拾う (基準線に有利側に倒した写像)。
GINZA_TO_SUMI: dict[str, PIIType] = {
    # --- 人名 ---
    "Person": PIIType.NAME,
    "Character": PIIType.NAME,
    # --- 住所を構成する地名系 ---
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
    # --- 日付 ---
    "Date": PIIType.DOB,
    "Era": PIIType.DOB,
    # --- 形式が決まっているもの ---
    "Phone_Number": PIIType.PHONE,
    "Email": PIIType.EMAIL,
    "Postal_Code": PIIType.POSTAL_CODE,
    "ID_Number": PIIType.MEMBER_ID,
    "Money": PIIType.MEMBER_ID,  # 使わない (下で除外)。写像の網羅性を示すためだけに残す
}
# Money は PII ではないので実際には落とす
GINZA_TO_SUMI.pop("Money", None)

#: 住所として連結してよい地名系ラベル (隣接時に1スパンへマージ)。
_ADDRESS_LABELS = {
    "Address", "Postal_Address", "Province", "City", "County",
    "Country", "GPE_Other", "Region_Other", "Domestic_Region",
}


class PresidioGinzaBaseline:
    """Presidio の形式レコグナイザ + GiNZA の日本語NER を合成した条件。

    Claim: 検出率 — 日本語NERを足せばどこまで届くのか、という
    実務上いちばん現実的な比較対象を提供する。
    """

    name = "presidio_ginza"
    label = "(B) Presidio + GiNZA"

    def __init__(self, join_address: bool = True) -> None:
        self.join_address = join_address
        self._nlp = None
        self._engine = None

    def info(self) -> BaselineInfo:
        """条件のメタ情報。

        Claim: CPU速度 — ja_ginza (electra 系 transformer を含まない bunsetu 版) は
        おおよそ 5e7 パラメータ規模。散布図の座標に用いる。
        """
        return BaselineInfo(
            name=self.name,
            label=self.label,
            params=5e7,
            runtime="spaCy/GiNZA + presidio regex",
            notes="GiNZA 拡張固有表現を Presidio エンティティへ写像し住所断片を連結",
        )

    def available(self) -> bool:
        """依存が揃っているか。

        Claim: 検出率 — 環境不備を検出漏れと誤認しないため。
        """
        try:
            import spacy
            from presidio_analyzer import AnalyzerEngine  # noqa: F401

            spacy.load("ja_ginza")
            return True
        except Exception:
            return False

    def warmup(self) -> None:
        """GiNZA と Presidio の形式レコグナイザを読み込む。

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
        self._nlp("ウォームアップ 田中太郎")
        self._engine.analyze(text="ウォームアップ", language="ja")

    def detect(self, text: str) -> list[Span]:
        """GiNZA の固有表現と Presidio の形式ルールを合成して返す。

        Claim: 検出率 / 低誤検出 — 氏名・住所は GiNZA、電話/メール等は Presidio、
        という実務でよく組まれる分担をそのまま再現する。
        """
        self.warmup()
        assert self._nlp is not None and self._engine is not None

        spans: list[Span] = []

        # --- GiNZA 固有表現 ---
        doc = self._nlp(text)
        ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
        if self.join_address:
            ents = _join_adjacent_address(ents, text)
        for a, b, lab in ents:
            t = GINZA_TO_SUMI.get(lab)
            if t is None or b <= a:
                continue
            spans.append(
                Span(
                    start=a, end=b, label=t, text=text[a:b],
                    score=0.85, source=Source.BASELINE,
                    meta={"entity_type": lab, "baseline": self.name, "layer": "ginza"},
                )
            )

        # --- Presidio の形式レコグナイザ ---
        from benchmarks.baselines.presidio_base import PRESIDIO_TO_SUMI

        for r in self._engine.analyze(text=text, language="ja"):
            t = PRESIDIO_TO_SUMI.get(r.entity_type)
            if t is None or r.end <= r.start:
                continue
            spans.append(
                Span(
                    start=r.start, end=r.end, label=t, text=text[r.start : r.end],
                    score=float(r.score), source=Source.BASELINE,
                    meta={"entity_type": r.entity_type, "baseline": self.name, "layer": "presidio"},
                )
            )

        return _resolve_overlaps(spans)


def _join_adjacent_address(
    ents: list[tuple[int, int, str]], text: str
) -> list[tuple[int, int, str]]:
    """隣接する地名系エンティティを1つの住所スパンへ連結する。

    Claim: 検出率 — GiNZA は「東京都新宿区西新宿」と「2-8-1」を別々に切りがちなので、
    間に区切り文字しか無い地名断片と、直後に続く丁目/番地表現を連結して、
    基準線が住所を最大限拾えるよう **有利に** 補正する。
    """
    import re

    out: list[tuple[int, int, str]] = []
    ents = sorted(ents, key=lambda e: e[0])
    i = 0
    while i < len(ents):
        a, b, lab = ents[i]
        if lab in _ADDRESS_LABELS:
            j = i + 1
            while j < len(ents):
                a2, b2, lab2 = ents[j]
                gap = text[b:a2]
                if lab2 in _ADDRESS_LABELS and len(gap) <= 1 and gap.strip("　 ") == "":
                    b = b2
                    j += 1
                else:
                    break
            # 直後に続く丁目/番地表現を取り込む
            m = re.match(r"[\s　]*((?:\d+[-\d]*)|(?:[一二三四五六七八九十]+丁目[\d一二三四五六七八九十番地号\-]*))",
                         text[b:])
            if m and m.group(1):
                b = b + m.end()
            out.append((a, b, "Address"))
            i = j
        else:
            out.append((a, b, lab))
            i += 1
    return out


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """重なるスパンを score 優先で解消し、非重複・start 昇順にする。

    Claim: 低誤検出 — 重なりを残したまま採点すると基準線の誤検出数が
    不当に膨らむため、重複は畳んでから比較する。
    """
    chosen: list[Span] = []
    for s in sorted(spans, key=lambda s: (-s.score, -(s.end - s.start), s.start)):
        if all(not s.overlaps(c) for c in chosen):
            chosen.append(s)
    return sorted(chosen, key=lambda s: (s.start, s.end))


if __name__ == "__main__":
    b = PresidioGinzaBaseline()
    print("available:", b.available())
    tests = [
        "田中太郎様よりご連絡をいただきました。連絡先は090-1234-5678、tanaka.taro@example.co.jp です。住所は東京都新宿区西新宿2-8-1、生年月日は1985年3月4日。",
        "森の中を歩いていると、林業の振興について泉が湧くように話が広がった。",
        "型番TX-2024-0355、注文番号0120-8834-221でお問い合わせください。",
        "長野県の気候は寒暖差が大きい。福島の復興も進んでいる。",
    ]
    for t in tests:
        print("\n" + t[:50])
        for s in b.detect(t):
            print(f"   {s.label.ja:8s} {s.slice_of(t)!r} ({s.meta.get('entity_type')})")
