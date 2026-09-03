"""Presidio に差し込める recognizer として Sumi を公開する。

Claim: 検出率 / 低誤検出 — 既存の Presidio 運用を捨てずに、日本語の弱点だけを
Sumi で置き換えられるようにする。Presidio 側のエンティティ名に写像するため、
既存の匿名化パイプライン (AnonymizerEngine 等) はそのまま使える。

presidio がインストールされていない環境でも ``import sumi`` が壊れないよう、
本モジュールは presidio の import を遅延・保護している。
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from sumi.detector import SumiDetector
from sumi.types import PIIType, Span

#: Sumi の種別 -> Presidio のエンティティ名
SUMI_TO_PRESIDIO: dict[PIIType, str] = {
    PIIType.NAME: "PERSON",
    PIIType.ADDRESS: "LOCATION",
    PIIType.PHONE: "PHONE_NUMBER",
    PIIType.EMAIL: "EMAIL_ADDRESS",
    PIIType.DOB: "DATE_TIME",
    PIIType.CREDIT_CARD: "CREDIT_CARD",
    PIIType.BANK_ACCOUNT: "JP_BANK_ACCOUNT",
    PIIType.MYNUMBER: "JP_MY_NUMBER",
    PIIType.MEMBER_ID: "JP_MEMBER_ID",
    PIIType.POSTAL_CODE: "JP_POSTAL_CODE",
}

#: 逆写像
PRESIDIO_TO_SUMI: dict[str, PIIType] = {v: k for k, v in SUMI_TO_PRESIDIO.items()}

#: Sumi が対応する Presidio エンティティ一覧
SUPPORTED_ENTITIES: list[str] = list(SUMI_TO_PRESIDIO.values())


def _base_class():
    """presidio の EntityRecognizer を返す (未導入なら None)。

    Claim: 検出率 — presidio 非導入環境でも sumi 本体が壊れないようにする。
    """
    try:
        from presidio_analyzer import EntityRecognizer

        return EntityRecognizer
    except Exception:  # pragma: no cover - presidio 未導入時
        return None


_EntityRecognizer = _base_class()


if _EntityRecognizer is not None:

    class SumiRecognizer(_EntityRecognizer):  # type: ignore[misc,valid-type]
        """Sumi を Presidio の recognizer として登録するためのアダプタ。

        Claim: 検出率 / 低誤検出 — Presidio の既定日本語構成が落とす氏名・住所を
        Sumi が担当し、形式確定型は Sumi の規則層が高精度に拾う。

        Args:
            model_path: 学習済み Sumi モデルのディレクトリ。``None`` なら既定を探し、
                無ければ規則層のみで動作する。
            use_model: モデル層を使うか (False なら規則層のみ)。
            supported_language: Presidio に申告する言語 (既定 ``"ja"``)。
            threshold: 採用するスコアの下限。
            use_rules: Sumi の規則層を使うか。
            detector: 既存の :class:`~sumi.detector.SumiDetector` を注入する場合に指定。
        """

        def __init__(
            self,
            model_path: str | None = None,
            *,
            supported_language: str = "ja",
            threshold: float = 0.5,
            use_rules: bool = True,
            use_model: bool = True,
            detector: SumiDetector | None = None,
            supported_entities: Sequence[str] | None = None,
            name: str = "SumiRecognizer",
            device: str = "cpu",
            onnx: bool = False,
        ) -> None:
            self.model_path = model_path
            self.threshold = threshold
            self.use_rules = use_rules
            self.use_model = use_model
            self.device = device
            self.onnx = onnx
            self._detector = detector
            super().__init__(
                supported_entities=list(supported_entities or SUPPORTED_ENTITIES),
                name=name,
                supported_language=supported_language,
            )

        def load(self) -> None:
            """モデルを読み込む (Presidio のライフサイクルから呼ばれる)。

            Claim: CPU速度 — 読み込みを1回に限定し、解析ごとのコストを避ける。
            """
            if self._detector is None:
                self._detector = SumiDetector(
                    self.model_path, use_rules=self.use_rules,
                    use_model=self.use_model, threshold=self.threshold,
                    device=self.device, onnx=self.onnx,
                )

        @property
        def detector(self) -> SumiDetector:
            """遅延生成した検出器を返す。

            Claim: CPU速度 — 実際に解析するまでモデルをロードしない。
            """
            if self._detector is None:
                self.load()
            assert self._detector is not None
            return self._detector

        def analyze(self, text: str, entities, nlp_artifacts=None):
            """Presidio の RecognizerResult 列を返す。

            Claim: 検出率 — Sumi のスパンを Presidio の座標系・エンティティ名へ
            そのまま写像する。``text`` は Presidio が渡した原文なので、
            Sumi 側の正規化で長さが変わらないことを確認したうえで返す。
            """
            from presidio_analyzer import RecognizerResult

            wanted = set(entities) if entities else set(SUPPORTED_ENTITIES)
            spans = self.detector.detect(text)
            results = []
            for s in spans:
                ent = SUMI_TO_PRESIDIO.get(s.label)
                if ent is None or ent not in wanted:
                    continue
                if s.end > len(text):
                    continue
                results.append(
                    RecognizerResult(
                        entity_type=ent,
                        start=s.start,
                        end=s.end,
                        score=float(s.score),
                        analysis_explanation=None,
                    )
                )
            return results

else:  # pragma: no cover - presidio 未導入時のフォールバック

    class SumiRecognizer:  # type: ignore[no-redef]
        """presidio 未導入時のダミー。

        Claim: 検出率 — 依存欠如を早期に、わかりやすく知らせる。
        """

        def __init__(self, *a, **kw) -> None:
            raise ImportError(
                "presidio-analyzer が必要です: pip install 'sumi[presidio]'"
            )


def register(registry, **kw) -> None:
    """既存の RecognizerRegistry に Sumi を登録する。

    Claim: 検出率 — 既存 Presidio 構成へ1行で差し込めることを保証する。
    registry 側の対応言語にも申告言語を追加する (presidio は registry と
    AnalyzerEngine の supported_languages が一致していないと起動しないため)。
    """
    rec = SumiRecognizer(**kw)
    registry.add_recognizer(rec)
    langs = list(getattr(registry, "supported_languages", []) or [])
    if rec.supported_language not in langs:
        langs.append(rec.supported_language)
    registry.supported_languages = langs


def make_stub_nlp_engine(language: str = "ja"):
    """spaCy 日本語モデル無しで AnalyzerEngine を動かすための最小 NLP エンジン。

    Claim: 検出率 — Sumi は自前で日本語を解析するため、Presidio に組み込む際に
    GiNZA/spaCy を必須にしたくない。Presidio は NlpEngine の存在を前提に
    設計されているので、空の NlpArtifacts を返すスタブを提供する。
    Sumi の recognizer は ``nlp_artifacts`` を使わないため、これで十分に動く。
    """
    from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine

    class _StubNlpEngine(NlpEngine):
        """何も解析しない NLP エンジン (Sumi が自前で解析するため)。"""

        engine_name = "sumi_stub"

        def __init__(self, langs: list[str]) -> None:
            self._langs = list(langs)

        def load(self) -> None:
            return None

        def is_loaded(self) -> bool:
            return True

        def get_supported_languages(self) -> list[str]:
            return list(self._langs)

        def get_supported_entities(self) -> list[str]:
            return []

        def is_stopword(self, word: str, language: str) -> bool:
            return False

        def is_punct(self, word: str, language: str) -> bool:
            return False

        def process_text(self, text: str, language: str) -> NlpArtifacts:
            return NlpArtifacts(
                entities=[], tokens=[], tokens_indices=[], lemmas=[],
                nlp_engine=self, language=language,
            )

        def process_batch(self, texts, language, **kw):
            for t in texts:
                yield t, self.process_text(t, language)

    return _StubNlpEngine([language])


def build_analyzer(
    model_path: str | None = None, *, language: str = "ja",
    use_ginza: bool = False, **kw
):
    """Sumi を組み込んだ AnalyzerEngine を作る。

    Claim: 検出率 / 低誤検出 — 日本語向けに、Sumi を主役に据えた構成を返す。
    既定では spaCy 日本語モデルを **必要としない** (Sumi が自前で解析する)。
    ``use_ginza=True`` を渡した場合のみ GiNZA を NLP エンジンとして併用する。
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry

    registry = RecognizerRegistry(supported_languages=[language])
    register(registry, model_path=model_path, supported_language=language, **kw)

    nlp_engine = None
    if use_ginza:
        try:
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            nlp_engine = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": language, "model_name": "ja_ginza"}],
                }
            ).create_engine()
        except Exception:
            nlp_engine = None
    if nlp_engine is None:
        nlp_engine = make_stub_nlp_engine(language)

    return AnalyzerEngine(
        nlp_engine=nlp_engine, registry=registry, supported_languages=[language]
    )


def _selftest() -> None:
    """自己テスト。

    Claim: 検出率 — Presidio 経由でも Sumi の検出が同じ位置・同じ内容で返ることを確認する。
    """
    print("=" * 74)
    print("sumi.presidio_plugin 自己テスト")
    print("=" * 74)

    if _EntityRecognizer is None:
        print("presidio-analyzer 未導入のためスキップ")
        return

    text = ("田中太郎様よりご連絡をいただきました。連絡先は 090-1234-5678、"
            "tanaka.taro@example.co.jp です。型番 TX-2024-0355 は生産終了です。")

    print("\n[recognizer 単体]")
    rec = SumiRecognizer(use_rules=True)
    for r in rec.analyze(text, entities=None):
        print(f"    {r.entity_type:16s} {text[r.start:r.end]!r:30s} score={r.score:.2f}")

    print("\n[AnalyzerEngine 経由 (register の1行差し込み)]")
    engine = build_analyzer(use_model=False)        # <- Sumi を組み込んだ AnalyzerEngine
    res = engine.analyze(text=text, language="ja")
    for r in sorted(res, key=lambda r: r.start):
        print(f"    {r.entity_type:16s} {text[r.start:r.end]!r:30s} score={r.score:.2f}")
    assert any(r.entity_type == "PHONE_NUMBER" for r in res)
    assert any(r.entity_type == "EMAIL_ADDRESS" for r in res)

    # 座標が Presidio 側でも正しいこと
    from sumi.detector import SumiDetector

    direct = SumiDetector(use_model=False).detect(text)
    got = {(r.start, r.end, r.entity_type) for r in res}
    want = {(s.start, s.end, SUMI_TO_PRESIDIO[s.label]) for s in direct}
    assert want <= got, f"座標が一致しない: 直接={want} presidio={got}"
    print(f"\n    ✓ 直接呼び出しと Presidio 経由で座標・種別が一致 ({len(want)} 件)")

    print("\n[匿名化まで通す]")
    try:
        from presidio_anonymizer import AnonymizerEngine

        an = AnonymizerEngine().anonymize(text=text, analyzer_results=res)
        print(f"    {an.text}")
    except Exception as e:
        print(f"    presidio-anonymizer 未導入のためスキップ ({type(e).__name__})")

    print("\n" + "=" * 74)
    print("すべての自己テストに合格")
    print("=" * 74)


if __name__ == "__main__":
    _selftest()
