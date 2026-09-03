"""(D)(E) Sumi — 量子化前 (PyTorch/CPU) と量子化後 (ONNX INT8/CPU)。

Claim: 検出率 / 低誤検出 / CPU速度 — 本命条件。他条件とまったく同じ
``detect(text) -> list[Span]`` に揃えて、同一の指標で採点される。
"""

from __future__ import annotations

import os

from benchmarks.baselines import BaselineInfo
from sumi.detector import DEFAULT_MODEL_DIR, SumiDetector
from sumi.types import Span


class SumiBaseline:
    """Sumi 検出器をベンチマーク条件として包む。

    Claim: 検出率 / CPU速度 — 配布物と同じ :class:`~sumi.detector.SumiDetector`
    をそのまま測ることで、「評価した構成」と「配る構成」の乖離を防ぐ。
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        onnx: bool = False,
        threshold: float = 0.5,
        use_calibration: bool = True,
        device: str = "cpu",
        use_model: bool = True,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        self.model_path = model_path or DEFAULT_MODEL_DIR
        self.onnx = onnx
        self.use_model = use_model
        self.threshold = threshold
        self.use_calibration = use_calibration
        self.device = device
        self.name = name or ("sumi_int8" if onnx else "sumi_fp32")
        self.label = label or (
            "(E) Sumi 量子化後 (ONNX INT8)" if onnx else "(D) Sumi 量子化前 (fp32)"
        )
        self._det: SumiDetector | None = None

    def info(self) -> BaselineInfo:
        """条件のメタ情報 (散布図の座標に使う)。

        Claim: CPU速度 — 0.13B というモデル規模を明示し、
        4B 級 LLM (条件 C) との対比を可能にする。
        """
        params = 132.4e6 if self.use_model else 0.0
        if self._det is not None and self._det.model is not None:
            params = sum(p.numel() for p in self._det.model.model.parameters())
        return BaselineInfo(
            name=self.name,
            label=self.label,
            params=params,
            runtime="onnxruntime (CPU)" if self.onnx else "PyTorch (CPU)",
            quantization="INT8 (dynamic)" if self.onnx else "none",
            notes="ModernBERT-Ja-130m + 規則層 + 較正",
        )

    def available(self) -> bool:
        """学習済みモデルが存在するか。

        Claim: 検出率 — 未学習を検出漏れと取り違えないため。
        """
        if not self.use_model:
            return True          # 規則層のみなら常に利用可能
        if not os.path.isdir(self.model_path):
            return False
        if self.onnx:
            return any(
                os.path.exists(os.path.join(self.model_path, n))
                for n in ("model.int8.onnx", "model.onnx")
            )
        return os.path.exists(os.path.join(self.model_path, "sumi_labels.json"))

    def warmup(self) -> None:
        """検出器を構築し、1回推論して定常状態にする。

        Claim: CPU速度 — ロードとグラフ構築を計測から除外する。
        """
        if self._det is not None:
            return
        calibrator = None
        cal = os.path.join(self.model_path, "calibrator.json")
        if self.use_calibration and os.path.exists(cal):
            from sumi.calibrate import SpanCalibrator

            calibrator = SpanCalibrator.load(cal)
        self._det = SumiDetector(
            self.model_path if self.use_model else None,
            use_model=self.use_model,
            threshold=self.threshold, calibrator=calibrator,
            device=self.device, onnx=self.onnx,
        )
        self._det.detect("ウォームアップ 田中太郎 090-1234-5678")

    def detect(self, text: str) -> list[Span]:
        """1文書を検出する。

        Claim: 検出率 / 低誤検出 — 規則層とモデル層を統合した最終出力。
        """
        self.warmup()
        assert self._det is not None
        return self._det.detect(text)

    def detect_batch(self, texts) -> list[list[Span]]:
        """バッチ検出 (速度計測用の本来の経路)。

        Claim: CPU速度 — 実運用に近いバッチ処理でのスループットを測る。
        """
        self.warmup()
        assert self._det is not None
        return self._det.detect_batch(list(texts))


if __name__ == "__main__":
    for onnx in (False, True):
        b = SumiBaseline(onnx=onnx)
        print(f"{b.label}: available={b.available()}")
        if b.available():
            t = "田中太郎様、連絡先は090-1234-5678、住所は東京都新宿区西新宿2-8-1です。型番TX-2024-0355。"
            for s in b.detect(t):
                print(f"    {s.label.ja:8s} {s.text!r}")
