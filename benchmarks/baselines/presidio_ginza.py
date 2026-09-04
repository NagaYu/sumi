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

# 写像と住所連結は sumi.compare を **唯一の実装** として参照する。
# ここに複製すると、Gradio デモが見せる比較とベンチマークが数字を出す比較が
# 静かに食い違いうるため。
from sumi.compare import (  # noqa: E402
    ADDRESS_LABELS as _ADDRESS_LABELS,
    GINZA_TO_SUMI,
    PresidioGinzaComparator,
    join_adjacent_address as _join_adjacent_address,
    resolve_overlaps as _resolve_overlaps,
)


class PresidioGinzaBaseline(PresidioGinzaComparator):
    """(B) 条件。検出そのものは :class:`sumi.compare.PresidioGinzaComparator` に委譲する。

    Claim: 検出率 — ベンチマークの (B) と Gradio デモの比較表示が
    同じコードを通ることを、継承によって保証する。
    """

    name = "presidio_ginza"
    label = "(B) Presidio + GiNZA"

    def info(self) -> BaselineInfo:
        """条件のメタ情報。

        Claim: CPU速度 — ja_ginza はおおよそ 5e7 パラメータ規模。散布図の座標に用いる。
        """
        return BaselineInfo(
            name=self.name,
            label=self.label,
            params=5e7,
            runtime="spaCy/GiNZA + presidio regex",
            notes="GiNZA 拡張固有表現を Presidio エンティティへ写像し住所断片を連結",
        )


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
