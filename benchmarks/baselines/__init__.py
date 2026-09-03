"""比較条件 (A)-(E) の共通インターフェース。

Claim: 検出率 / 低誤検出 / CPU速度 — すべての条件を同じ ``detect(text) -> list[Span]``
に揃えることで、精度も速度も同一の物差しで比較できるようにする。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sumi.types import Span


@runtime_checkable
class Baseline(Protocol):
    """比較条件が満たすべき最小インターフェース。"""

    name: str
    label: str

    def available(self) -> bool: ...
    def warmup(self) -> None: ...
    def detect(self, text: str) -> list[Span]: ...


@dataclass
class BaselineInfo:
    """条件ごとのメタ情報 (図表の注記に使う)。

    Claim: CPU速度 — モデル規模とランタイムを記録し、
    「モデルサイズ vs 精度」散布図の座標を後から再構成できるようにする。
    """

    name: str
    label: str
    params: float | None = None       # パラメータ数 (10億=1e9 単位ではなく実数)
    runtime: str = ""                 # "spaCy" / "llama.cpp" / "PyTorch" / "onnxruntime"
    quantization: str = "none"
    notes: str = ""
    extra: dict = field(default_factory=dict)


def timed_detect(baseline: Baseline, texts: list[str]) -> tuple[list[list[Span]], float]:
    """検出を実行し、総経過時間 (秒) を返す。

    Claim: CPU速度 — ウォームアップを除いた純粋な推論時間だけを計測し、
    条件間の速度比較を公平にする。
    """
    out: list[list[Span]] = []
    t0 = time.perf_counter()
    for t in texts:
        out.append(baseline.detect(t))
    return out, time.perf_counter() - t0
