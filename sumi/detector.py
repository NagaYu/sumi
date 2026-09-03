"""SumiDetector — 規則層・モデル層・較正を束ねる公開ファサード。

Claim: 検出率 / 低誤検出 / CPU速度 / 可逆性 — Sumi を1つの API から使えるようにする。
CLI・Gradio・Presidio プラグイン・ベンチマークはすべてこの層を経由するため、
「評価した構成」と「配布する構成」が食い違わないことが構造的に保証される。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from sumi.mask import MaskMap, ReversibleMasker
from sumi.rules import RuleLayer, merge_spans
from sumi.types import RULE_DETERMINISTIC, PIIType, Source, Span, normalize

DEFAULT_MODEL_DIR = "artifacts/sumi-model"


@dataclass
class DetectResult:
    """1文書分の検出結果。

    Claim: 検出率 / CPU速度 — スパンと一緒に層別の所要時間を返すことで、
    精度と速度を同じ呼び出しから観測できるようにする。
    """

    text: str
    spans: list[Span] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON 化可能な辞書に変換する。

        Claim: 検出率 — CLI/API の出力形式を1か所に定める。
        """
        return {
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
            "timings": {k: round(v, 6) for k, v in self.timings.items()},
        }


class SumiDetector:
    """規則層 + モデル層 + 較正を統合した検出器。

    Claim: 検出率 / 低誤検出 — 形式確定型は規則で確実に、文脈依存型はモデルで。
    統合順序は :func:`sumi.rules.merge_spans` の明示的な5段階に従う。

    Args:
        model_path: 学習済みモデルのディレクトリ。``None`` なら既定を探し、
            見つからなければ **規則層のみ** で動作する (モデル未学習でも使える)。
        use_rules: 規則層を使うか。
        use_model: モデル層を使うか。
        threshold: モデルスパンの採用閾値。**較正後**のスコアに対して適用する。
        model_threshold: モデル層から取り出す際の下限 (既定 0.05)。較正前に
            切り捨てると較正が意味を失い、閾値スイープも下方向に探索できなくなるため、
            ここは低く保ち、採否は較正後に1度だけ判断する。
        calibrator: :class:`sumi.calibrate.SpanCalibrator`。あればスコアを較正する。
        device: ``"cpu"`` / ``"mps"``。ベンチマークの既定は ``"cpu"``。
        onnx: True なら ONNX Runtime を用いる (量子化後条件 (E) で使用)。
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        use_rules: bool = True,
        use_model: bool = True,
        threshold: float = 0.5,
        calibrator=None,
        device: str = "cpu",
        onnx: bool = False,
        max_length: int = 256,
        batch_size: int = 16,
        model_threshold: float = 0.05,
    ) -> None:
        self.threshold = threshold
        # モデル層から取り出す際の下限。**較正の前** に切ってはいけないので、
        # ここは十分に低くし、採否の判断は較正後の self.threshold で1度だけ行う。
        # これにより (a) 較正がスコアを持ち上げた低スコアスパンを拾えるようになり、
        # (b) ベンチマークの閾値スイープが self.threshold より下も探索できる。
        self.model_threshold = min(model_threshold, threshold)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.calibrator = calibrator
        self.use_rules = use_rules
        self.onnx = onnx

        self.rules = RuleLayer() if use_rules else None

        self.model = None
        self._ort = None
        self.model_path = model_path
        if use_model:
            path = model_path or (DEFAULT_MODEL_DIR if os.path.isdir(DEFAULT_MODEL_DIR) else None)
            if path and os.path.isdir(path):
                self.model_path = path
                if onnx:
                    self._load_onnx(path)
                else:
                    from sumi.model import TokenClassifier

                    self.model = TokenClassifier.load(path, device=device)
        self.use_model = self.model is not None or self._ort is not None

    # ------------------------------------------------------------------ onnx
    def _load_onnx(self, path: str) -> None:
        """ONNX Runtime セッションと tokenizer を読み込む。

        Claim: CPU速度 — 量子化済みモデルを CPU で動かす経路 (条件 (E))。
        """
        import onnxruntime as ort
        from transformers import AutoTokenizer

        cand = [os.path.join(path, n) for n in ("model.int8.onnx", "model.onnx")]
        onnx_path = next((c for c in cand if os.path.exists(c)), None)
        if onnx_path is None:
            raise FileNotFoundError(f"ONNX model not found under {path}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("SUMI_ORT_THREADS", "0")) or 0
        self._ort = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self._ort_path = onnx_path
        self._tok = AutoTokenizer.from_pretrained(path)
        with open(os.path.join(path, "sumi_labels.json"), encoding="utf-8") as f:
            self._labels = json.load(f)["label_list"]

    def _onnx_predict(self, texts: Sequence[str]) -> list[list[Span]]:
        """ONNX 経路での推論 (スライディングウィンドウなし・単純版)。

        Claim: CPU速度 — 量子化モデルの実効スループットを測るための最短経路。
        """
        from sumi.model import decode_bio, refine_boundaries

        out: list[list[Span]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc = self._tok(
                batch, return_offsets_mapping=True, truncation=True,
                max_length=self.max_length, padding=True, return_tensors="np",
            )
            logits = self._ort.run(
                None,
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )[0]
            probs = _softmax(logits, axis=-1)
            for j, text in enumerate(batch):
                offs = [tuple(map(int, o)) for o in enc["offset_mapping"][j]]
                spans = decode_bio(probs[j], offs, text, self._labels,
                                   threshold=self.model_threshold)
                out.append([refine_boundaries(s, text) for s in spans])
        return out

    # --------------------------------------------------------------- predict
    def _model_spans(self, texts: Sequence[str]) -> list[list[Span]]:
        """モデル層の生スパンを返す。

        Claim: 検出率 — 文脈依存の氏名・住所・生年月日を担当する層。
        """
        if self._ort is not None:
            return self._onnx_predict(texts)
        if self.model is None:
            return [[] for _ in texts]
        return self.model.predict(
            list(texts), batch_size=self.batch_size,
            max_length=self.max_length, threshold=self.model_threshold, refine=True,
        )

    def _calibrate(self, spans: list[Span]) -> list[Span]:
        """較正器があればモデルスパンのスコアを較正する。

        Claim: 較正 — 出力スコアを「そのスパンが真である確率」に近づけ、
        閾値の意味を運用者に理解可能なものにする。
        """
        if self.calibrator is None or not spans:
            return spans
        model_spans = [s for s in spans if s.source is not Source.RULE]
        if not model_spans:
            return spans
        cal = self.calibrator.transform([s.score for s in model_spans])
        it = iter(cal)
        return [
            s if s.source is Source.RULE else s.with_(score=float(next(it)))
            for s in spans
        ]

    def detect(self, text: str) -> list[Span]:
        """1文書を検出する。

        Claim: 検出率 / 低誤検出 — 規則とモデルを明示的優先順位で統合して返す。
        """
        return self.detect_result(text).spans

    def detect_result(self, text: str) -> DetectResult:
        """検出結果を層別の所要時間つきで返す。

        Claim: CPU速度 — 規則層とモデル層それぞれの寄与を切り分けて計測する。
        """
        text = normalize(text)
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        rule_spans = self.rules.detect(text) if self.rules else []
        timings["rules"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        model_spans = self._model_spans([text])[0] if self.use_model else []
        timings["model"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        model_spans = self._calibrate(model_spans)
        model_spans = [s for s in model_spans if s.score >= self.threshold]
        spans = merge_spans(model_spans, rule_spans, text, rule_types=RULE_DETERMINISTIC)
        timings["merge"] = time.perf_counter() - t0
        timings["total"] = sum(timings.values())
        return DetectResult(text=text, spans=spans, timings=timings)

    def detect_batch(self, texts: Sequence[str]) -> list[list[Span]]:
        """複数文書をまとめて検出する (モデル推論をバッチ化)。

        Claim: CPU速度 — 1件ずつ呼ぶより実効スループットが上がる経路を明示する。
        """
        texts = [normalize(t) for t in texts]
        rule_all = [self.rules.detect(t) if self.rules else [] for t in texts]
        model_all = self._model_spans(texts) if self.use_model else [[] for _ in texts]
        out: list[list[Span]] = []
        for text, r, m in zip(texts, rule_all, model_all):
            m = self._calibrate(m)
            m = [s for s in m if s.score >= self.threshold]
            out.append(merge_spans(m, r, text, rule_types=RULE_DETERMINISTIC))
        return out

    # ---------------------------------------------------------------- redact
    def redact(
        self, text: str, *, masker: ReversibleMasker | None = None, doc_id: str = ""
    ) -> tuple[str, MaskMap]:
        """検出して墨消し (可逆マスク) したテキストと対応表を返す。

        Claim: 可逆性 — 検出からマスクまでを1呼び出しに閉じ、
        スパン座標の受け渡しミスによる復元不能を構造的に防ぐ。
        """
        text = normalize(text)
        spans = self.detect(text)
        m = masker or ReversibleMasker()
        return m.mask(text, spans, doc_id=doc_id)

    def info(self) -> dict:
        """現在の構成を返す (再現性のための記録)。

        Claim: CPU速度 / 検出率 — ベンチマーク結果がどの構成で得られたかを
        結果ファイルに残せるようにする。
        """
        return {
            "model_path": self.model_path,
            "use_rules": bool(self.rules),
            "use_model": bool(self.use_model),
            "onnx": bool(self._ort is not None),
            "device": self.device,
            "threshold": self.threshold,
            "model_threshold": self.model_threshold,
            "calibrated": self.calibrator is not None,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数値安定な softmax。

    Claim: 較正 — ONNX 経路でも PyTorch 経路と同じ確率スケールを得る。
    """
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _selftest() -> None:
    """自己テスト。

    Claim: 検出率 / 低誤検出 / 可逆性 — 規則のみでも動くこと、
    学習済みモデルがあれば統合されること、墨消しが可逆であることを確認する。
    """
    print("=" * 74)
    print("sumi.detector 自己テスト")
    print("=" * 74)

    text = ("田中太郎様よりご連絡をいただきました。連絡先は 090-1234-5678、"
            "tanaka.taro@example.co.jp です。住所は東京都新宿区西新宿2-8-1。"
            "なお型番 TX-2024-0355 は生産終了です。")

    print("\n[規則層のみ] (モデル未学習でも動作すること)")
    d = SumiDetector(use_model=False)
    r = d.detect_result(text)
    for s in r.spans:
        print(f"    {s.label.ja:8s} {s.text!r:30s} score={s.score:.2f} from={s.meta.get('from')}")
    print(f"    timings: {({k: round(v*1000,2) for k,v in r.timings.items()})} ms")
    assert any(s.label is PIIType.PHONE for s in r.spans)
    assert any(s.label is PIIType.EMAIL for s in r.spans)
    assert not any("TX-2024" in s.text for s in r.spans), "型番を拾ってしまっている"

    print("\n[可逆マスク]")
    masked, mmap = d.redact(text)
    print(f"    {masked}")
    restored = ReversibleMasker().unmask(masked, mmap)
    assert restored == normalize(text), "復元が原文に一致しない"
    print("    ✓ 完全復元")

    if os.path.isdir(DEFAULT_MODEL_DIR):
        print(f"\n[統合] 学習済みモデル {DEFAULT_MODEL_DIR} を検出")
        dm = SumiDetector(DEFAULT_MODEL_DIR, device="cpu")
        for s in dm.detect(text):
            print(f"    {s.label.ja:8s} {s.text!r:30s} from={s.meta.get('from')}")
        print(f"    info: {dm.info()}")
    else:
        print(f"\n[統合] {DEFAULT_MODEL_DIR} が無いため規則層のみで検証 "
              f"(scripts/train.py 実行後に再確認)")

    print("\n" + "=" * 74)
    print("すべての自己テストに合格")
    print("=" * 74)


if __name__ == "__main__":
    _selftest()
