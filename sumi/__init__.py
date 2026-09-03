"""Sumi — 日本語PII検出。CPUで動く小型モデルと規則層、可逆マスキング。

Claim: 検出率 / 低誤検出 / CPU速度 / 可逆性 — パッケージの入口。
重い依存 (torch/transformers/presidio) は遅延 import し、
``import sumi`` だけなら規則層と型定義だけで完結するようにしている。

    >>> from sumi import SumiDetector
    >>> det = SumiDetector()              # 学習済みモデルが無ければ規則層のみ
    >>> masked, mapping = det.redact("連絡先は090-1234-5678です。")

**法令遵守を保証するものではありません。** 個人情報が外部へ出るリスクを
下げる道具であり、完全な検出器ではありません。検出漏れを前提に運用してください。
"""

from __future__ import annotations

__version__ = "0.1.0"

from sumi.types import (  # noqa: F401
    ALL_TYPES,
    MODEL_DRIVEN,
    RULE_DETERMINISTIC,
    Document,
    PIIType,
    Source,
    Span,
    bio_labels,
    normalize,
    spans_to_bio,
)

__all__ = [
    "__version__",
    # types
    "PIIType", "Source", "Span", "Document",
    "ALL_TYPES", "RULE_DETERMINISTIC", "MODEL_DRIVEN",
    "normalize", "bio_labels", "spans_to_bio",
    # lazy
    "SumiDetector", "DetectResult", "RuleLayer", "merge_spans",
    "ReversibleMasker", "MaskMap", "LLMRoundTrip",
    "EgressGuard", "EgressViolation",
    "TokenClassifier", "TrainConfig",
    "SpanCalibrator", "recall_at_fixed_fpr",
    "HardNegativeGenerator", "PIIFactory",
]

#: 属性名 -> (モジュール, 名前) の遅延 import 表
_LAZY = {
    "SumiDetector": ("sumi.detector", "SumiDetector"),
    "DetectResult": ("sumi.detector", "DetectResult"),
    "RuleLayer": ("sumi.rules", "RuleLayer"),
    "merge_spans": ("sumi.rules", "merge_spans"),
    "ReversibleMasker": ("sumi.mask", "ReversibleMasker"),
    "MaskMap": ("sumi.mask", "MaskMap"),
    "LLMRoundTrip": ("sumi.mask", "LLMRoundTrip"),
    "EgressGuard": ("sumi.egress", "EgressGuard"),
    "EgressViolation": ("sumi.egress", "EgressViolation"),
    "TokenClassifier": ("sumi.model", "TokenClassifier"),
    "TrainConfig": ("sumi.model", "TrainConfig"),
    "SpanCalibrator": ("sumi.calibrate", "SpanCalibrator"),
    "recall_at_fixed_fpr": ("sumi.calibrate", "recall_at_fixed_fpr"),
    "HardNegativeGenerator": ("sumi.negatives", "HardNegativeGenerator"),
    "PIIFactory": ("sumi.synth", "PIIFactory"),
}


def __getattr__(name: str):
    """重い依存を持つシンボルを遅延 import する。

    Claim: CPU速度 — ``import sumi`` の時点で torch を読み込まないことで、
    規則層だけを使う軽量な用途 (CLI の一部・Presidio 連携) の起動を速く保つ。
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'sumi' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


def __dir__() -> list[str]:
    """補完候補を返す。

    Claim: CPU速度 — 遅延 import でも対話環境で見えるようにする。
    """
    return sorted(__all__)
