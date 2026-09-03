"""スパン確信度の較正 (calibration) と、本プロジェクトの評価指標の定義。

このモジュールは Sumi 全体の「数字の出どころ」である。README・図表・ベンチマークが
引用する指標はすべてここで定義され、ここ以外で再定義してはならない。

指標の定義 (README と図の脚注はこの定義をそのまま引く):

* **検出率 (recall)** — 正解スパンのうち、予測スパンと突き合わせできた割合。
  突き合わせは :func:`match_spans` の ``exact`` (start/end/label 完全一致) か
  ``partial`` (同一 label かつ 1 文字以上の重なり) を明示して用いる。
* **誤検出率** — 否定例文書に対して出てしまった予測スパンの量。3 つの言い方があり
  混同されやすいので :func:`false_positive_report` で 3 つとも返す。
  README と図が「誤検出率」と呼ぶのは **文書レベル誤検出率**
  (誤検出が 1 件以上出た否定例文書の割合) である。
* **主要指標 recall@fixed-FPR** — 誤検出率を実務で許せる水準に固定したときの検出率。
  :func:`recall_at_fixed_fpr` を参照。
* **較正 (calibration)** — スコアを「そのスパンが本当に PII である確率」として
  読めるようにすること。:class:`SpanCalibrator` と
  :func:`expected_calibration_error` が担当する。

Claim: 較正 / 検出率 / 低誤検出 — 指標の定義そのものを 1 箇所に閉じ込めることで、
主張の数値が「どう測ったか」まで再現可能になる。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .types import ALL_TYPES, Document, PIIType, Source, Span

__all__ = [
    "SpanCalibrator",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "reliability_diagram",
    "use_japanese_font",
    "JP_FONT_CANDIDATES",
    "match_spans",
    "detection_rates",
    "false_positive_rate",
    "false_positive_report",
    "recall_at_fixed_fpr",
    "MATCH_MODES",
    "FPR_METRICS",
]

# --------------------------------------------------------------------------------------
# 小道具
# --------------------------------------------------------------------------------------

_EPS = 1e-6
#: 突き合わせモード。
MATCH_MODES: tuple[str, ...] = ("exact", "partial")
#: 誤検出率の数え方 (:func:`recall_at_fixed_fpr` の ``fpr_metric``)。
FPR_METRICS: tuple[str, ...] = ("per_doc", "doc_level", "per_1000_chars")


def _as_float_array(x: Any) -> np.ndarray:
    """入力を 1 次元 float64 配列にそろえる (内部用)。

    Claim: 較正 — list / ndarray / スカラの混在で指標がずれないようにする。
    """
    a = np.asarray(list(x) if isinstance(x, (list, tuple)) else x, dtype=np.float64)
    return np.atleast_1d(a).ravel()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """数値的に安全なロジスティック関数 (内部用)。

    Claim: 較正 — 温度スケーリングの前後で overflow を起こさないため。
    """
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logit(p: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """確率をロジットへ (0/1 は eps でクリップする) (内部用)。

    Claim: 較正 — 0 と 1 に張り付いたスコアでも温度が推定できるようにする。
    """
    q = np.clip(p, eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def _minimize_scalar_bounded(fn, lo: float, hi: float, *, tol: float = 1e-8) -> float:
    """scipy が無い環境向けの黄金分割探索 (内部用フォールバック)。

    Claim: 較正 — 依存が欠けても温度較正が動くことを保証する。
    """
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(200):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def _pava(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-Adjacent-Violators による単調回帰 (sklearn が無い場合の内部実装)。

    Claim: 較正 — isotonic 較正を標準ライブラリ + numpy だけでも再現できるようにする。
    """
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order].astype(np.float64)
    # 同一 x はまとめる
    uniq_x, inv = np.unique(xs, return_inverse=True)
    sums = np.zeros(uniq_x.size)
    cnts = np.zeros(uniq_x.size)
    np.add.at(sums, inv, ys)
    np.add.at(cnts, inv, 1.0)
    vals = sums / cnts
    # スタックで単調化 (v=ブロック平均, w=重み合計, k=ブロックに含まれる uniq_x の個数)
    v_stack: list[float] = []
    w_stack: list[float] = []
    k_stack: list[int] = []
    for v, wi in zip(vals, cnts):
        v_cur, w_cur, k_cur = float(v), float(wi), 1
        while v_stack and v_stack[-1] > v_cur:
            v_prev = v_stack.pop()
            w_prev = w_stack.pop()
            k_cur += k_stack.pop()
            w_new = w_prev + w_cur
            v_cur = (v_prev * w_prev + v_cur * w_cur) / w_new
            w_cur = w_new
        v_stack.append(v_cur)
        w_stack.append(w_cur)
        k_stack.append(k_cur)
    out = np.empty(uniq_x.size, dtype=np.float64)
    i = 0
    for v, k in zip(v_stack, k_stack):
        out[i : i + k] = v
        i += k
    return uniq_x, np.clip(out, 0.0, 1.0)


def _compress_knots(x: np.ndarray, y: np.ndarray, *, max_knots: int = 512) -> tuple[list[float], list[float]]:
    """単調関数の折れ点を、形を保ったまま間引く (内部用)。

    Claim: 較正 — 較正器を JSON で保存できる大きさに保つ (isotonic も JSON 一本で済む)。
    """
    if x.size == 0:
        return [], []
    keep = [0]
    for i in range(1, x.size - 1):
        if not math.isclose(y[i], y[keep[-1]], rel_tol=0.0, abs_tol=1e-9):
            keep.append(i)
    if x.size > 1:
        keep.append(x.size - 1)
    keep = sorted(set(keep))
    if len(keep) > max_knots:
        sel = np.unique(np.linspace(0, len(keep) - 1, max_knots).round().astype(int))
        keep = [keep[i] for i in sel]
    return [float(x[i]) for i in keep], [float(y[i]) for i in keep]


# --------------------------------------------------------------------------------------
# 較正器
# --------------------------------------------------------------------------------------


class SpanCalibrator:
    """スパンの生スコアを「そのスパンが真である確率」へ写す較正器。

    2 つの方式を持つ。

    * ``"temperature"`` — スパンのロジットを 1 個のスカラ温度 ``T`` で割る。
      ``T > 1`` で自信を弱め、``T < 1`` で強める。パラメータが 1 個なので
      検証データが少なくても過学習しない。
    * ``"isotonic"`` — 単調回帰。任意の単調な歪みを直せるが、データを食う。

    どちらも ``transform`` は最終的に「折れ線 (knots) を線形補間」または
    「ロジット/温度」という単純な形で表現され、JSON 1 ファイルで保存・復元できる。

    Claim: 較正 — 閾値 0.5 が「五分五分」を意味するようにスコアを直し、
    運用側が確率としてスコアを読めるようにする。
    """

    #: 実装済みの較正方式。
    METHODS: tuple[str, ...] = ("temperature", "isotonic")
    #: 保存形式のバージョン。
    VERSION: str = "1"

    def __init__(self, method: str = "temperature", *, eps: float = _EPS,
                 bounds: tuple[float, float] = (1e-2, 1e2)) -> None:
        """較正方式を選んで初期化する。

        Args:
            method: ``"temperature"`` または ``"isotonic"``。
            eps: ロジット変換時のクリップ幅。
            bounds: 温度 ``T`` の探索範囲 (下限 > 0)。

        Claim: 較正 — 方式を明示的に選ばせ、既定は最も壊れにくい温度スケーリングにする。
        """
        if method not in self.METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {self.METHODS}")
        if not (0.0 < bounds[0] < bounds[1]):
            raise ValueError(f"invalid temperature bounds: {bounds}")
        self.method = method
        self.eps = float(eps)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.temperature_: float | None = None
        self.knots_x_: list[float] | None = None
        self.knots_y_: list[float] | None = None
        self.n_fit_: int = 0
        self.meta: dict[str, Any] = {}

    # ---- 状態 -------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """学習済みかどうか。

        Claim: 較正 — 未学習の較正器を推論経路に挿してしまう事故を検出可能にする。
        """
        if self.method == "temperature":
            return self.temperature_ is not None
        return bool(self.knots_x_)

    def __repr__(self) -> str:  # pragma: no cover - 表示のみ
        if self.method == "temperature":
            t = "unfitted" if self.temperature_ is None else f"T={self.temperature_:.4f}"
        else:
            t = "unfitted" if not self.knots_x_ else f"knots={len(self.knots_x_)}"
        return f"SpanCalibrator(method={self.method!r}, {t}, n_fit={self.n_fit_})"

    # ---- 学習 -------------------------------------------------------------------

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "SpanCalibrator":
        """検証集合のスコアと正解 (0/1) から較正関数を推定する。

        温度スケーリングは負の対数尤度 (NLL) を ``scipy.optimize.minimize_scalar``
        (無ければ黄金分割探索) で最小化する。isotonic は
        ``sklearn.isotonic.IsotonicRegression`` (無ければ内蔵 PAVA) を使い、
        結果を折れ線の折れ点として保持する。

        Args:
            scores: 生スコア (0..1 を想定。範囲外はクリップ)。
            labels: 正解ラベル。1 = そのスパンは真、0 = 誤検出。

        Returns:
            自分自身 (メソッドチェーン用)。

        Claim: 較正 — 学習時の分布とは別の検証集合でスコアを補正することで、
        ECE (期待較正誤差) を下げる。
        """
        s = np.clip(_as_float_array(scores), 0.0, 1.0)
        y = _as_float_array(labels)
        if s.size == 0:
            raise ValueError("cannot fit a calibrator on an empty score list")
        if s.size != y.size:
            raise ValueError(f"scores/labels length mismatch: {s.size} != {y.size}")
        if np.any((y < -1e-9) | (y > 1.0 + 1e-9)):
            raise ValueError("labels must be 0/1 (or probabilities in [0,1])")
        y = np.clip(y, 0.0, 1.0)
        self.n_fit_ = int(s.size)
        pos = float(y.sum())
        self.meta = {
            "n": int(s.size),
            "n_positive": pos,
            "base_rate": pos / s.size,
            "degenerate_labels": bool(pos <= 0.0 or pos >= s.size),
        }

        if self.method == "temperature":
            z = _logit(s, self.eps)

            def nll(t: float) -> float:
                p = np.clip(_sigmoid(z / max(t, 1e-12)), self.eps, 1.0 - self.eps)
                return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

            lo, hi = self.bounds
            best: float
            try:  # scipy があればそちらを使う
                from scipy.optimize import minimize_scalar  # type: ignore

                res = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                                      options={"xatol": 1e-8})
                best = float(res.x)
                self.meta["optimizer"] = "scipy.minimize_scalar(bounded)"
                self.meta["nll"] = float(res.fun)
            except Exception:  # pragma: no cover - scipy は環境にある想定
                best = float(_minimize_scalar_bounded(nll, lo, hi))
                self.meta["optimizer"] = "golden-section"
                self.meta["nll"] = nll(best)
            self.temperature_ = float(min(max(best, lo), hi))
            self.knots_x_ = None
            self.knots_y_ = None
            self.meta["nll_before"] = nll(1.0)
        else:
            xs: np.ndarray
            ys: np.ndarray
            try:
                from sklearn.isotonic import IsotonicRegression  # type: ignore

                iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                         out_of_bounds="clip")
                iso.fit(s, y)
                xs = np.asarray(getattr(iso, "X_thresholds_", np.unique(s)), dtype=np.float64)
                ys = np.asarray(getattr(iso, "y_thresholds_", iso.predict(np.unique(s))),
                                dtype=np.float64)
                self.meta["backend"] = "sklearn.isotonic"
            except Exception:  # pragma: no cover - sklearn は環境にある想定
                xs, ys = _pava(s, y)
                self.meta["backend"] = "builtin-pava"
            kx, ky = _compress_knots(np.asarray(xs, dtype=np.float64),
                                     np.clip(np.asarray(ys, dtype=np.float64), 0.0, 1.0))
            if not kx:  # 全部同じ x → 定数関数
                kx, ky = [float(s[0])], [float(y.mean())]
            self.knots_x_ = kx
            self.knots_y_ = ky
            self.temperature_ = None
            self.meta["n_knots"] = len(kx)
        return self

    # ---- 適用 -------------------------------------------------------------------

    def transform(self, scores: Sequence[float] | float) -> list[float]:
        """生スコアを較正済み確率へ写す。

        Args:
            scores: スカラでも列でもよい。

        Returns:
            0..1 に収まる確率の list (入力がスカラでも長さ 1 の list)。

        Claim: 較正 — 推論経路で 1 行差し込むだけで、以後のスコアを確率として扱える。
        """
        if not self.fitted:
            raise RuntimeError("SpanCalibrator.transform() called before fit()/load()")
        s = np.clip(_as_float_array(scores), 0.0, 1.0)
        if self.method == "temperature":
            t = float(self.temperature_ or 1.0)
            out = _sigmoid(_logit(s, self.eps) / max(t, 1e-12))
        else:
            kx = np.asarray(self.knots_x_, dtype=np.float64)
            ky = np.asarray(self.knots_y_, dtype=np.float64)
            if kx.size == 1:
                out = np.full_like(s, float(ky[0]))
            else:
                out = np.interp(s, kx, ky, left=float(ky[0]), right=float(ky[-1]))
        return [float(v) for v in np.clip(out, 0.0, 1.0)]

    def transform_spans(self, spans: Iterable[Span]) -> list[Span]:
        """スパン列の ``score`` を較正済み確率に差し替えた新しい列を返す。

        元のスコアは ``meta["score_raw"]`` に残す (再現性のため)。

        Claim: 較正 / 可逆性 — スパンを不変のまま差し替えるので、
        較正の前後で座標がずれずマスク・復元と整合する。
        """
        out: list[Span] = []
        spans = list(spans)
        if not spans:
            return out
        cal = self.transform([sp.score for sp in spans])
        for sp, c in zip(spans, cal):
            meta = dict(sp.meta)
            meta.setdefault("score_raw", float(sp.score))
            meta["calibrator"] = self.method
            out.append(sp.with_(score=float(c), meta=meta))
        return out

    def nll(self, scores: Sequence[float], labels: Sequence[int]) -> float:
        """較正後スコアの負の対数尤度 (小さいほど良い)。

        Claim: 較正 — ECE と併せて報告することで、
        「よく当たる」と「確率として正しい」を区別して示せる。
        """
        p = np.clip(_as_float_array(self.transform(scores)), self.eps, 1.0 - self.eps)
        y = np.clip(_as_float_array(labels), 0.0, 1.0)
        if p.size != y.size:
            raise ValueError("scores/labels length mismatch")
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    # ---- 保存・復元 ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON 化できる辞書へ (保存形式そのもの)。

        Claim: 較正 — 較正器を人が読めるテキストで持ち回れるようにする
        (pickle を使わないので配布・レビューが容易)。
        """
        return {
            "version": self.VERSION,
            "method": self.method,
            "eps": self.eps,
            "bounds": list(self.bounds),
            "temperature": None if self.temperature_ is None else float(self.temperature_),
            "knots_x": list(self.knots_x_ or []),
            "knots_y": list(self.knots_y_ or []),
            "n_fit": int(self.n_fit_),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SpanCalibrator":
        """:meth:`to_dict` の出力から復元する。

        Claim: 較正 — 保存 → 復元で ``transform`` が完全に一致することを保証する。
        """
        obj = cls(str(d.get("method", "temperature")),
                  eps=float(d.get("eps", _EPS)),
                  bounds=tuple(d.get("bounds", (1e-2, 1e2))))  # type: ignore[arg-type]
        t = d.get("temperature", None)
        obj.temperature_ = None if t is None else float(t)
        kx = list(d.get("knots_x") or [])
        ky = list(d.get("knots_y") or [])
        obj.knots_x_ = [float(v) for v in kx] or None
        obj.knots_y_ = [float(v) for v in ky] or None
        obj.n_fit_ = int(d.get("n_fit", 0))
        obj.meta = dict(d.get("meta") or {})
        return obj

    def save(self, path: str | os.PathLike[str]) -> None:
        """較正器を JSON として保存する (temperature / isotonic 共通)。

        Claim: 較正 — 学習済みモデルとは独立に較正だけ差し替え・再配布できる。
        """
        p = Path(path)
        if p.parent and str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "SpanCalibrator":
        """:meth:`save` で書いた JSON から較正器を読み込む。

        Claim: 較正 — 推論側は較正の学習手順を知らなくても結果だけ使える。
        """
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d)


# --------------------------------------------------------------------------------------
# 較正の指標と図
# --------------------------------------------------------------------------------------


def _bin_stats(scores: Sequence[float], labels: Sequence[int], bins: int):
    """等幅ビンごとの (件数, 平均スコア, 実測正解率) を返す (内部用)。

    Claim: 較正 — ECE と信頼性ダイアグラムが必ず同じビン分割を見るようにする。
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    s = np.clip(_as_float_array(scores), 0.0, 1.0)
    y = np.clip(_as_float_array(labels), 0.0, 1.0)
    if s.size != y.size:
        raise ValueError(f"scores/labels length mismatch: {s.size} != {y.size}")
    edges = np.linspace(0.0, 1.0, bins + 1)
    if s.size == 0:
        z = np.zeros(bins)
        return edges, z, z.copy(), z.copy()
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, bins - 1)
    counts = np.zeros(bins, dtype=np.float64)
    conf = np.zeros(bins, dtype=np.float64)
    acc = np.zeros(bins, dtype=np.float64)
    for b in range(bins):
        m = idx == b
        c = float(m.sum())
        counts[b] = c
        if c:
            conf[b] = float(s[m].mean())
            acc[b] = float(y[m].mean())
    return edges, counts, conf, acc


def expected_calibration_error(scores: Sequence[float], labels: Sequence[int],
                               bins: int = 15) -> float:
    """期待較正誤差 (ECE)。等幅ビンでの |実測正解率 - 平均スコア| の加重平均。

    ``ECE = Σ_b (n_b / N) * |acc_b - conf_b|`` (空ビンは寄与 0)。
    ビンは ``[0,1]`` の等幅 ``bins`` 分割で、スコア 1.0 は最終ビンに入れる。

    Args:
        scores: 予測スコア (較正前でも後でもよい)。
        labels: そのスパンが真なら 1、誤検出なら 0。
        bins: ビン数 (既定 15)。

    Returns:
        0 に近いほど「スコアが確率として正しい」。

    Claim: 較正 — 較正の前後を 1 つの数字で比較できるようにする。
    """
    _, counts, conf, acc = _bin_stats(scores, labels, bins)
    n = float(counts.sum())
    if n <= 0:
        return 0.0
    return float(np.sum(counts / n * np.abs(acc - conf)))


def maximum_calibration_error(scores: Sequence[float], labels: Sequence[int],
                              bins: int = 15) -> float:
    """最大較正誤差 (MCE)。ビンごとのずれの最大値 (件数 0 のビンは無視)。

    Claim: 較正 — 平均で埋もれる「特定スコア帯だけ大きく外している」を見つける。
    """
    _, counts, conf, acc = _bin_stats(scores, labels, bins)
    m = counts > 0
    if not np.any(m):
        return 0.0
    return float(np.max(np.abs(acc[m] - conf[m])))


def brier_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Brier スコア (二乗誤差)。較正と識別性能を合わせた総合指標。

    Claim: 較正 — ECE がビン分割に依存するのに対し、こちらは分割に依らない対照指標。
    """
    s = np.clip(_as_float_array(scores), 0.0, 1.0)
    y = np.clip(_as_float_array(labels), 0.0, 1.0)
    if s.size != y.size:
        raise ValueError("scores/labels length mismatch")
    if s.size == 0:
        return 0.0
    return float(np.mean((s - y) ** 2))


#: 日本語 (CJK) フォント候補。先頭から順に探す。
JP_FONT_CANDIDATES: tuple[str, ...] = (
    "Hiragino Sans",
    "Hiragino Maru Gothic ProN",
    "Hiragino Kaku Gothic ProN",
    "Apple SD Gothic Neo",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAPGothic",
    "TakaoPGothic",
    "YuGothic",
    "MS Gothic",
)

_FONT_CACHE: dict[str, str | None] = {}


def _pyplot():
    """matplotlib.pyplot を遅延 import する (内部用)。

    Claim: CPU速度 — 検出経路では matplotlib を読み込まないので起動が速い。
    """
    import matplotlib

    if os.environ.get("MPLBACKEND") is None:
        try:
            matplotlib.use("Agg")
        except Exception:  # pragma: no cover
            pass
    import matplotlib.pyplot as plt  # noqa: E402

    return plt


def use_japanese_font(*, candidates: Sequence[str] = JP_FONT_CANDIDATES,
                      apply: bool = True) -> str | None:
    """利用できる日本語 (CJK) フォントを 1 つ選び、matplotlib に設定する。

    ``figures.py`` と benchmarks からも import される共通ヘルパ。
    候補を順に ``matplotlib.font_manager`` で探し、最初に見つかったものを
    ``rcParams["font.family"]`` に設定する。**1 つも無くても例外を投げず
    ``None`` を返す** ので、呼び出し側は英語ラベルに落とせばよい。

    Args:
        candidates: 探索するフォント名 (先頭優先)。
        apply: True なら rcParams を書き換える。False なら探すだけ。

    Returns:
        見つかったフォント名。見つからなければ ``None``。

    Claim: 較正 — 信頼性ダイアグラム等の図が、日本語フォントの無い CI/他マシンでも
    必ず生成できる (図が出ないと較正の主張が示せない)。
    """
    key = f"{apply}:{','.join(candidates)}"
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    found: str | None = None
    try:
        import matplotlib
        from matplotlib import font_manager

        available = set()
        try:
            available = {f.name for f in font_manager.fontManager.ttflist}
        except Exception:  # pragma: no cover
            available = set()
        for name in candidates:
            if name in available:
                found = name
                break
            try:  # ttc など list に出ないものへの保険
                font_manager.findfont(font_manager.FontProperties(family=name),
                                      fallback_to_default=False)
                found = name
                break
            except Exception:
                continue
        if apply:
            matplotlib.rcParams["axes.unicode_minus"] = False
            if found:
                base = list(matplotlib.rcParams.get("font.sans-serif", []))
                matplotlib.rcParams["font.family"] = "sans-serif"
                matplotlib.rcParams["font.sans-serif"] = [found] + [
                    b for b in base if b != found
                ]
    except Exception:  # pragma: no cover - matplotlib すら無い場合
        found = None
    _FONT_CACHE[key] = found
    return found


def reliability_diagram(scores: Sequence[float], labels: Sequence[int], *,
                        bins: int = 15, title: str = "",
                        out_path: str | None = None):
    """信頼性ダイアグラム (対角線・ビン別正解率・件数ヒストグラム・ECE 注記) を描く。

    日本語フォントが無い環境では :func:`use_japanese_font` が ``None`` を返すため、
    ラベルを英語に自動で切り替える (**落ちない**)。

    Args:
        scores: 予測スコア。
        labels: 1 = 真、0 = 誤検出。
        bins: 等幅ビン数 (ECE と同じ分割)。
        title: 図のタイトル (空なら既定文言)。
        out_path: 指定すると PNG として保存する (親ディレクトリは自動生成)。

    Returns:
        ``matplotlib.figure.Figure``。

    Claim: 較正 — 「較正した」という主張を、対角線からのずれとして目で確認できる形にする。
    """
    plt = _pyplot()
    font = use_japanese_font()
    ja = font is not None

    def L(ja_text: str, en_text: str) -> str:
        return ja_text if ja else en_text

    edges, counts, conf, acc = _bin_stats(scores, labels, bins)
    ece = expected_calibration_error(scores, labels, bins)
    mce = maximum_calibration_error(scores, labels, bins)
    n = int(counts.sum())
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = (1.0 / bins) * 0.9
    nz = counts > 0

    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=140)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1.2,
            label=L("完全較正", "Perfect calibration"), zorder=5)
    ax.bar(centers[nz], acc[nz], width=width, color="#3b6ea5", edgecolor="#20364f",
           linewidth=0.7, label=L("実測正解率", "Empirical accuracy"), zorder=3)
    gap = conf - acc
    ax.bar(centers[nz], gap[nz], bottom=acc[nz], width=width, color="#d9534f",
           alpha=0.35, edgecolor="#8c2f2c", linewidth=0.7, hatch="///",
           label=L("ずれ (スコア - 正解率)", "Gap (score - accuracy)"), zorder=4)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(L("予測スコア (確信度)", "Predicted score (confidence)"))
    ax.set_ylabel(L("実測正解率", "Empirical accuracy"))
    ax.set_title(title or L("信頼性ダイアグラム", "Reliability diagram"))
    ax.grid(alpha=0.25, linestyle=":", zorder=0)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.text(
        0.985, 0.055,
        L(f"ECE = {ece:.4f}\nMCE = {mce:.4f}\nN = {n}",
          f"ECE = {ece:.4f}\nMCE = {mce:.4f}\nN = {n}"),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#999999",
                  alpha=0.95),
        zorder=6,
    )

    # 件数ヒストグラム (インセット)
    try:
        ins = ax.inset_axes([0.545, 0.315, 0.42, 0.24])
        ins.bar(centers, counts, width=width, color="#7f8c8d", edgecolor="none")
        ins.set_title(L("ビン別件数", "Samples per bin"), fontsize=7.5, pad=2.0)
        ins.set_xlim(0.0, 1.0)
        ins.tick_params(labelsize=6.0, length=2.0)
        ins.set_yticks([0, int(counts.max())] if counts.max() > 0 else [0])
        for spine in ("top", "right"):
            ins.spines[spine].set_visible(False)
        ins.patch.set_alpha(0.85)
    except Exception:  # pragma: no cover - 古い matplotlib への保険
        pass

    fig.tight_layout()
    if out_path:
        p = Path(out_path)
        if p.parent and str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------------------
# スパンの突き合わせと検出率
# --------------------------------------------------------------------------------------


def _spans_of(obj: Any) -> list[Span]:
    """Document / list[Span] / dict のいずれからでもスパン列を取り出す (内部用)。

    Claim: 検出率 — 評価入口を寛容にして、呼び出し側の型ゆれで指標が落ちないようにする。
    """
    if obj is None:
        return []
    if isinstance(obj, Document):
        return list(obj.spans)
    if isinstance(obj, Span):
        return [obj]
    if isinstance(obj, Mapping):
        return [s if isinstance(s, Span) else Span.from_dict(s) for s in obj.get("spans", [])]
    if hasattr(obj, "spans"):
        return list(obj.spans)
    return [s if isinstance(s, Span) else Span.from_dict(s) for s in obj]


def _doc_id_of(obj: Any, i: int) -> str:
    """文書 ID を取り出す (無ければ連番) (内部用)。

    Claim: 検出率 — 予測を dict で渡された場合の突き合わせキーにする。
    """
    if isinstance(obj, Document):
        return obj.doc_id or f"#{i}"
    if isinstance(obj, Mapping):
        return str(obj.get("doc_id") or f"#{i}")
    return str(getattr(obj, "doc_id", "") or f"#{i}")


def _text_of(obj: Any) -> str:
    """本文を取り出す (無ければ空文字) (内部用)。

    Claim: 低誤検出 — 1000 文字あたり誤検出率の分母を得るため。
    """
    if isinstance(obj, Document):
        return obj.text
    if isinstance(obj, Mapping):
        return str(obj.get("text", ""))
    return str(getattr(obj, "text", "") or "")


def _align(docs: Sequence[Any], preds: Any) -> tuple[list[Any], list[list[Span]]]:
    """文書列と予測列を対応づける (list でも doc_id 辞書でも可) (内部用)。

    Claim: 検出率 — ずれた対応づけで検出率が水増しされる事故を防ぐ。
    """
    dl = list(docs)
    if isinstance(preds, Mapping):
        out = [list(preds.get(_doc_id_of(d, i), [])) for i, d in enumerate(dl)]
    else:
        out = [list(p) for p in preds]
        if len(out) != len(dl):
            raise ValueError(f"docs/preds length mismatch: {len(dl)} docs vs {len(out)} preds")
    return dl, out


def match_spans(gold: list[Span], pred: list[Span], *,
                mode: str = "exact") -> tuple[list[tuple[Span, Span]], list[Span], list[Span]]:
    """正解スパンと予測スパンを 1 対 1 に突き合わせる。

    * ``mode="exact"`` — ``(start, end, label)`` が完全一致するものだけを対応させる。
    * ``mode="partial"`` — ``label`` が同じで文字区間が 1 文字以上重なれば候補。
      候補が競合する場合は **IoU の高い組から** 貪欲に確定させるので、
      1 つの正解が消費できる予測は 1 つだけ (二重計上が起きない)。
      同 IoU の場合はスコアの高い予測を優先し、最後は位置で決定論的に決める。

    Args:
        gold: 正解スパン列。
        pred: 予測スパン列。
        mode: ``"exact"`` または ``"partial"``。

    Returns:
        ``(tp_pairs, fp, fn)``。``tp_pairs`` は ``(gold_span, pred_span)`` の list、
        ``fp`` は対応が付かなかった予測、``fn`` は取りこぼした正解 (どちらも位置順)。

    Claim: 検出率 / 低誤検出 — 検出率と誤検出の数え方をここ 1 箇所に固定する。
    """
    if mode not in MATCH_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MATCH_MODES}")
    g = list(gold)
    p = list(pred)
    if not g or not p:
        return [], sorted(p, key=lambda s: (s.start, s.end)), sorted(g, key=lambda s: (s.start, s.end))

    cands: list[tuple[float, float, int, int, int, int]] = []
    for gi, gs in enumerate(g):
        for pi, ps in enumerate(p):
            if gs.label != ps.label:
                continue
            if mode == "exact":
                if gs.key() != ps.key():
                    continue
                iou = 1.0
            else:
                if not gs.overlaps(ps):
                    continue
                iou = gs.iou(ps)
                if iou <= 0.0:
                    continue
            cands.append((-iou, -float(ps.score), gs.start, ps.start, gi, pi))
    cands.sort()

    used_g: set[int] = set()
    used_p: set[int] = set()
    tp: list[tuple[Span, Span]] = []
    for _niou, _nscore, _gstart, _pstart, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tp.append((g[gi], p[pi]))
    tp.sort(key=lambda pair: (pair[0].start, pair[0].end))
    fp = sorted((s for i, s in enumerate(p) if i not in used_p), key=lambda s: (s.start, s.end))
    fn = sorted((s for i, s in enumerate(g) if i not in used_g), key=lambda s: (s.start, s.end))
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """precision / recall / f1 を計算する (0 除算は 0.0) (内部用)。

    Claim: 検出率 — 分母 0 の扱いを 1 箇所に固定し、表の値がぶれないようにする。
    """
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def detection_rates(gold_docs: Sequence[Any], pred_per_doc: Any, *,
                    mode: str = "partial") -> dict[str, Any]:
    """種別ごと + 全体の precision / recall / f1 を計算する。

    ``micro`` は全種別の TP/FP/FN を合算してから比を取る (件数の多い種別に引っ張られる)。
    ``macro`` は **正解が 1 件以上ある種別だけ** の単純平均 (少数種別も等価に扱う)。

    Args:
        gold_docs: :class:`~sumi.types.Document` の列 (spans が正解)。
        pred_per_doc: 文書ごとの予測スパン列 (list of list、または doc_id → list)。
        mode: :func:`match_spans` の突き合わせモード。

    Returns:
        ``{"by_type": {...}, "micro": {...}, "macro": {...}, "n_docs": int, "mode": str}``

    Claim: 検出率 — 「どの種別が弱いか」を種別別に開示する。全体値だけでは
    種別ごとの取りこぼしが平均に隠れてしまう。
    """
    docs, preds = _align(gold_docs, pred_per_doc)
    counts: dict[str, dict[str, int]] = {}

    def bucket(label: PIIType) -> dict[str, int]:
        return counts.setdefault(label.value, {"tp": 0, "fp": 0, "fn": 0})

    for d, ps in zip(docs, preds):
        tp, fp, fn = match_spans(_spans_of(d), list(ps), mode=mode)
        for g, _ in tp:
            bucket(g.label)["tp"] += 1
        for s in fp:
            bucket(s.label)["fp"] += 1
        for s in fn:
            bucket(s.label)["fn"] += 1

    by_type: dict[str, dict[str, float]] = {}
    for t in ALL_TYPES:
        c = counts.get(t.value)
        if not c:
            continue
        row = _prf(c["tp"], c["fp"], c["fn"])
        row.update({"tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                    "support": c["tp"] + c["fn"], "ja": t.ja})
        by_type[t.value] = row
    for key, c in counts.items():  # ALL_TYPES に無い label が来た場合の保険
        if key not in by_type:
            row = _prf(c["tp"], c["fp"], c["fn"])
            row.update({"tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                        "support": c["tp"] + c["fn"], "ja": key})
            by_type[key] = row

    mtp = sum(c["tp"] for c in counts.values())
    mfp = sum(c["fp"] for c in counts.values())
    mfn = sum(c["fn"] for c in counts.values())
    micro = _prf(mtp, mfp, mfn)
    micro.update({"tp": mtp, "fp": mfp, "fn": mfn, "support": mtp + mfn})

    sup = [r for r in by_type.values() if r["support"] > 0]
    if sup:
        macro = {k: float(np.mean([r[k] for r in sup])) for k in ("precision", "recall", "f1")}
    else:
        macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    macro["n_types"] = len(sup)

    return {"by_type": by_type, "micro": micro, "macro": macro,
            "n_docs": len(docs), "mode": mode}


# --------------------------------------------------------------------------------------
# 誤検出率
# --------------------------------------------------------------------------------------


def _fp_stats(neg_docs: Sequence[Any], pred_per_doc: Any, *,
              ignore_gold_matches: bool = False, mode: str = "partial") -> dict[str, Any]:
    """否定例側の誤検出集計 (3 つの誤検出率の共通土台) (内部用)。

    Claim: 低誤検出 — 3 つの定義が必ず同じ数え方から導かれることを保証する。
    """
    docs, preds = _align(neg_docs, pred_per_doc)
    n_docs = len(docs)
    n_chars = sum(len(_text_of(d)) for d in docs)
    n_fp = 0
    n_docs_with_fp = 0
    by_type: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for d, ps in zip(docs, preds):
        ps = list(ps)
        if ignore_gold_matches:
            _tp, fps, _fn = match_spans(_spans_of(d), ps, mode=mode)
        else:
            fps = ps
        if fps:
            n_docs_with_fp += 1
        n_fp += len(fps)
        for s in fps:
            key = s.label.value if isinstance(s.label, PIIType) else str(s.label)
            by_type[key] = by_type.get(key, 0) + 1
        for k in (getattr(d, "negative_kinds", None) or []):
            if fps:
                by_kind[k] = by_kind.get(k, 0) + len(fps)
    return {
        "n_docs": n_docs,
        "n_chars": n_chars,
        "n_fp": n_fp,
        "n_docs_with_fp": n_docs_with_fp,
        "by_type": by_type,
        "by_negative_kind": by_kind,
    }


def false_positive_rate(neg_docs: Sequence[Any], pred_per_doc: Any, *,
                        ignore_gold_matches: bool = False) -> float:
    """否定例文書 1 件あたりの誤検出スパン数 (0.0 が理想)。

    **定義**: ``(否定例文書に出た予測スパンの総数) / (否定例文書数)``。
    否定例文書は正解スパンを持たない前提なので、出た予測はすべて誤検出である。
    (``ignore_gold_matches=True`` にすると、正解を持つ文書でも一致分を除外できるが、
    README が引く既定の定義は上式そのままである。)

    他の 2 つの言い方 (1000 文字あたり、文書レベル) は
    :func:`false_positive_report` が同時に返す。

    Args:
        neg_docs: 否定例文書の列。
        pred_per_doc: 文書ごとの予測スパン列。
        ignore_gold_matches: True なら正解と一致した予測を誤検出から除く。

    Returns:
        1 文書あたりの期待誤検出件数。否定例が 0 件なら 0.0。

    Claim: 低誤検出 — 「1 文書処理するたびに平均何件の空振りを人が確認するか」
    という運用コストに直結する量。
    """
    st = _fp_stats(neg_docs, pred_per_doc, ignore_gold_matches=ignore_gold_matches)
    if st["n_docs"] == 0:
        return 0.0
    return float(st["n_fp"]) / float(st["n_docs"])


def false_positive_report(neg_docs: Sequence[Any], pred_per_doc: Any, *,
                          ignore_gold_matches: bool = False) -> dict[str, Any]:
    """誤検出率を **3 つの定義すべて** で返す (混同を避けるための正本)。

    返す率の定義:

    1. ``fp_per_doc`` — 誤検出スパン数 / 否定例文書数。:func:`false_positive_rate` と同値。
       文書長に依存するので、文書長が違うデータ間の比較には向かない。
    2. ``fp_per_1000_chars`` — 誤検出スパン数 / (総文字数 / 1000)。
       文書長で正規化した密度。長文コーパスとの比較に使う。
    3. ``doc_level_fp_rate`` — 誤検出が **1 件以上** 出た否定例文書の割合 (0..1)。
       **README と図が「誤検出率」と呼ぶのはこれ**。「この文書は人手確認が要るか」
       という運用上の判断単位に一致するため。

    Returns:
        上記 3 つの率と、その分子分母 (``n_fp`` / ``n_docs`` / ``n_chars`` /
        ``n_docs_with_fp``)、種別別内訳 ``by_type``、否定例種別別内訳
        ``by_negative_kind`` を含む dict。

    Claim: 低誤検出 — 誤検出率という言葉の 3 通りの意味を同時に開示し、
    数字の見かけを良くする定義の取り違えを防ぐ。
    """
    st = _fp_stats(neg_docs, pred_per_doc, ignore_gold_matches=ignore_gold_matches)
    n_docs = st["n_docs"]
    n_chars = st["n_chars"]
    out = dict(st)
    out["fp_per_doc"] = (st["n_fp"] / n_docs) if n_docs else 0.0
    out["fp_per_1000_chars"] = (st["n_fp"] / (n_chars / 1000.0)) if n_chars else 0.0
    out["doc_level_fp_rate"] = (st["n_docs_with_fp"] / n_docs) if n_docs else 0.0
    out["definitions"] = {
        "fp_per_doc": "誤検出スパン数 / 否定例文書数",
        "fp_per_1000_chars": "誤検出スパン数 / (総文字数 / 1000)",
        "doc_level_fp_rate": "誤検出が1件以上出た否定例文書の割合 (= README の誤検出率)",
    }
    return out


# --------------------------------------------------------------------------------------
# 主要指標: 誤検出率を固定したときの検出率
# --------------------------------------------------------------------------------------


def _filter_by_threshold(preds: Sequence[Sequence[Span]], t: float) -> list[list[Span]]:
    """閾値以上のスパンだけ残す (``score >= t``) (内部用)。

    Claim: 低誤検出 — 閾値の向き (以上/超過) を 1 箇所に固定する。
    """
    return [[s for s in ps if float(s.score) >= t] for ps in preds]


def recall_at_fixed_fpr(gold_docs: Sequence[Any], pred_per_doc_scored: Any,
                        neg_docs: Sequence[Any], neg_pred_per_doc_scored: Any, *,
                        target_fpr: float = 0.05, mode: str = "partial",
                        by_type: bool = True, fpr_metric: str = "per_doc",
                        max_curve_points: int = 256) -> dict[str, Any]:
    """**主要指標**。誤検出率を ``target_fpr`` 以下に固定したときの検出率。

    手順:

    1. 陽性側・否定例側の全予測スコアを集めて閾値候補を作る
       (候補の最大値より大きい番兵も 1 つ加える = 「何も出さない」設定)。
    2. 誤検出率は閾値に対し単調非増加なので、
       ``fpr(t) <= target_fpr`` を満たす **最小の** 閾値を二分探索する
       (最小の閾値 = 検出率が最大になる閾値)。
    3. その閾値での陽性側の検出率を全体・種別ごとに返す。

    ``fpr_metric`` で誤検出率の数え方を選ぶ (:data:`FPR_METRICS`)。
    ``"per_doc"`` (既定) は 1 文書あたりの誤検出スパン数、
    ``"doc_level"`` は誤検出が出た文書の割合 (README の「誤検出率」)、
    ``"per_1000_chars"`` は 1000 文字あたりの件数。
    ``target_fpr`` の単位はこの選択に従う。

    退化ケース:

    * 否定例が 0 件 → 誤検出率は常に 0 とみなし、最小閾値 (= 全採用) を返す。
      ``degenerate="no_negatives"``。
    * 予測が 1 件も無い → 閾値 0.0、検出率 0。``degenerate="no_predictions"``。
    * 全スコアが同値 → 候補は「全採用」と「何も出さない」の 2 通りだけ。
    * 目標が達成不能 (負の目標など) → 最も誤検出率が小さい閾値を返し
      ``achieved=False`` を立てる。
    * 目標が「何も出さない」でしか達成できない → ``achieved=True`` だが
      ``degenerate="only_empty_predictions_meet_target"`` を立てる (検出率 0)。

    Returns:
        ``{"threshold", "fpr", "overall_recall", "by_type", "curve", ...}``。
        ``curve`` は ``(threshold, fpr, recall)`` の三つ組の list (閾値昇順) で、
        ベンチマークが運用曲線としてそのまま描ける。

    Claim: 低誤検出 / 検出率 — 「誤検出を実務で許せる水準に固定したときに、
    どれだけ拾えるか」という運用上の問いに直接答える指標。
    """
    if mode not in MATCH_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MATCH_MODES}")
    if fpr_metric not in FPR_METRICS:
        raise ValueError(f"unknown fpr_metric {fpr_metric!r}; expected one of {FPR_METRICS}")

    pos_docs, pos_preds = _align(gold_docs, pred_per_doc_scored)
    ndocs, neg_preds = _align(neg_docs, neg_pred_per_doc_scored)

    n_gold = sum(len(_spans_of(d)) for d in pos_docs)
    all_scores = [float(s.score) for ps in pos_preds for s in ps]
    all_scores += [float(s.score) for ps in neg_preds for s in ps]

    degenerate: str | None = None
    if not all_scores:
        degenerate = "no_predictions"
        grid = [0.0]
    else:
        grid = sorted(set(round(v, 12) for v in all_scores))
        grid.append(float(max(grid) + 1e-6))  # 番兵: これ以上なら 1 件も採用されない

    fpr_cache: dict[float, float] = {}
    rec_cache: dict[float, float] = {}

    def fpr_at(t: float) -> float:
        if t in fpr_cache:
            return fpr_cache[t]
        if not ndocs:
            v = 0.0
        else:
            rep = false_positive_report(ndocs, _filter_by_threshold(neg_preds, t))
            v = float(rep[{"per_doc": "fp_per_doc",
                           "doc_level": "doc_level_fp_rate",
                           "per_1000_chars": "fp_per_1000_chars"}[fpr_metric]])
        fpr_cache[t] = v
        return v

    def recall_at(t: float) -> float:
        if t in rec_cache:
            return rec_cache[t]
        if n_gold == 0:
            v = 0.0
        else:
            res = detection_rates(pos_docs, _filter_by_threshold(pos_preds, t), mode=mode)
            v = float(res["micro"]["recall"])
        rec_cache[t] = v
        return v

    if not ndocs:
        degenerate = degenerate or "no_negatives"

    # --- 二分探索 (fpr は t に対し単調非増加なので述語は単調) -----------------------
    lo, hi = 0, len(grid) - 1
    achieved = True
    if fpr_at(grid[hi]) > target_fpr:
        # 何も出さなくても目標を満たせない (target < 0 など)
        chosen = min(range(len(grid)), key=lambda i: (fpr_at(grid[i]), grid[i]))
        achieved = False
        degenerate = degenerate or "target_unreachable"
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if fpr_at(grid[mid]) <= target_fpr:
                hi = mid
            else:
                lo = mid + 1
        chosen = lo
        if chosen == len(grid) - 1 and len(grid) > 1 and fpr_at(grid[0]) > target_fpr:
            degenerate = degenerate or "only_empty_predictions_meet_target"

    thr = float(grid[chosen])
    kept_pos = _filter_by_threshold(pos_preds, thr)
    det = detection_rates(pos_docs, kept_pos, mode=mode)
    neg_rep = false_positive_report(ndocs, _filter_by_threshold(neg_preds, thr)) if ndocs else {
        "fp_per_doc": 0.0, "fp_per_1000_chars": 0.0, "doc_level_fp_rate": 0.0,
        "n_fp": 0, "n_docs": 0, "n_docs_with_fp": 0, "n_chars": 0,
        "by_type": {}, "by_negative_kind": {},
    }

    # --- 運用曲線 -----------------------------------------------------------------
    if len(grid) <= max_curve_points:
        pts = list(range(len(grid)))
    else:
        sel = np.unique(np.linspace(0, len(grid) - 1, max_curve_points).round().astype(int))
        pts = sorted(set(sel.tolist()) | {0, len(grid) - 1, chosen})
    curve = [(float(grid[i]), float(fpr_at(grid[i])), float(recall_at(grid[i]))) for i in pts]

    by_type_out: dict[str, Any] = {}
    if by_type:
        for k, row in det["by_type"].items():
            by_type_out[k] = {
                "recall": row["recall"],
                "precision": row["precision"],
                "f1": row["f1"],
                "tp": row["tp"],
                "fp": row["fp"],
                "fn": row["fn"],
                "support": row["support"],
                "ja": row.get("ja", k),
            }

    return {
        "threshold": thr,
        "fpr": float(neg_rep[{"per_doc": "fp_per_doc",
                              "doc_level": "doc_level_fp_rate",
                              "per_1000_chars": "fp_per_1000_chars"}[fpr_metric]]),
        "overall_recall": float(det["micro"]["recall"]),
        "by_type": by_type_out,
        "curve": curve,
        # --- 以下は補助情報 (契約の必須キーは上の 5 つ) ---
        "target_fpr": float(target_fpr),
        "fpr_metric": fpr_metric,
        "achieved": bool(achieved),
        "degenerate": degenerate,
        "mode": mode,
        "overall_precision": float(det["micro"]["precision"]),
        "overall_f1": float(det["micro"]["f1"]),
        "macro_recall": float(det["macro"]["recall"]),
        "n_gold_spans": int(n_gold),
        "n_pos_docs": len(pos_docs),
        "n_neg_docs": len(ndocs),
        "n_kept_pos_spans": int(sum(len(p) for p in kept_pos)),
        "fp_report": {k: neg_rep[k] for k in
                      ("fp_per_doc", "fp_per_1000_chars", "doc_level_fp_rate",
                       "n_fp", "n_docs", "n_docs_with_fp")},
    }


# --------------------------------------------------------------------------------------
# 自己テスト
# --------------------------------------------------------------------------------------


def _selftest_calibration(seed: int = 0, n: int = 4000) -> dict[str, Any]:
    """既知の歪み (ロジットを 0.4 倍 = 0.5 側へ潰す) を作り、較正で戻せるか見る。

    Claim: 較正 — 「ECE が下がる」という主張を、正解が分かっている合成データで検証する。
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.5, size=n)               # 真のロジット
    p_true = _sigmoid(z)                            # 真の確率
    y = (rng.random(n) < p_true).astype(np.float64)  # 観測ラベル
    squash = 0.4
    s = _sigmoid(z * squash)                        # モデルの申告スコア (自信不足)
    # 学習用/検証用に分割 (較正は検証集合で当てはめ、別集合で評価する)
    half = n // 2
    fit_s, fit_y = s[:half], y[:half]
    ev_s, ev_y = s[half:], y[half:]

    out: dict[str, Any] = {
        "true_temperature": squash,
        "ece_before": expected_calibration_error(ev_s, ev_y, 15),
        "brier_before": brier_score(ev_s, ev_y),
        "scores": {"eval": ev_s, "labels": ev_y},
    }
    for method in ("temperature", "isotonic"):
        cal = SpanCalibrator(method).fit(fit_s, fit_y)
        p = cal.transform(ev_s)
        out[method] = {
            "cal": cal,
            "ece": expected_calibration_error(p, ev_y, 15),
            "brier": brier_score(p, ev_y),
            "nll": cal.nll(ev_s, ev_y),
            "probs": p,
        }
    return out


def _toy_doc(doc_id: str, items: Sequence[tuple[PIIType, str]], *, subset: str,
             filler: str = "これは合成の検証用本文であり実在の情報を含まない。") -> Document:
    """位置を記録しながら合成トークンを差し込んだ玩具文書を作る (自己テスト用)。

    Claim: 検出率 — 正解座標を構成的に持つ文書でしか、検出率の検算はできない。
    """
    parts: list[str] = []
    spans: list[Span] = []
    cur = 0
    for label, token in items:
        parts.append(filler)
        cur += len(filler)
        spans.append(Span(cur, cur + len(token), label, token, score=1.0, source=Source.GOLD))
        parts.append(token)
        cur += len(token)
    parts.append("以上。")
    doc = Document(text="".join(parts), spans=spans, doc_id=doc_id, subset=subset,
                   genre="selftest", source_license="synthetic (CC0-1.0)")
    doc.validate()
    return doc


def _selftest_metrics() -> dict[str, Any]:
    """手計算できる玩具データで match/検出率/誤検出率/主要指標を検算する。

    Claim: 検出率 / 低誤検出 — 指標の実装が定義どおりであることを、
    暗算できる小さな例で固定する (回帰テストの土台)。
    """
    # --- 陽性側: 正解 6 件、予測 6 件 (スコアを 1 件ずつずらす) ------------------
    gold_items = [
        [(PIIType.NAME, "甲野花子"), (PIIType.PHONE, "090-0000-0000")],
        [(PIIType.NAME, "乙川次郎"), (PIIType.EMAIL, "a@example.invalid")],
        [(PIIType.ADDRESS, "架空県架空市9-9-9"), (PIIType.POSTAL_CODE, "999-9999")],
    ]
    pos_docs = [_toy_doc(f"pos{i}", it, subset="test") for i, it in enumerate(gold_items)]
    pos_scores = [[0.99, 0.95], [0.85, 0.75], [0.65, 0.45]]
    pos_preds: list[list[Span]] = []
    for d, row in zip(pos_docs, pos_scores):
        pred = []
        for g, sc in zip(d.sorted_spans(), row):
            pred.append(Span(g.start, g.end, g.label, g.text, score=sc, source=Source.MERGED))
        pos_preds.append(pred)

    # --- 否定例側: 5 文書、誤検出 5 件 -------------------------------------------
    neg_docs = [Document(text=f"否定例文書{i}: 紛らわしい表現だけを含む合成文である。",
                         spans=[], doc_id=f"neg{i}", subset="negatives",
                         negative_kinds=["common_noun_surname"]) for i in range(5)]
    neg_scores = [0.9, 0.7, 0.6, 0.5, 0.3]
    neg_preds: list[list[Span]] = []
    for d, sc in zip(neg_docs, neg_scores):
        neg_preds.append([Span(0, 3, PIIType.NAME, d.text[0:3], score=sc, source=Source.MODEL)])

    return {"pos_docs": pos_docs, "pos_preds": pos_preds,
            "neg_docs": neg_docs, "neg_preds": neg_preds}


if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 78)
    print("sumi.calibrate 自己テスト")
    print("=" * 78)

    # ---------------------------------------------------------------- 1. 較正
    st = _selftest_calibration()
    ev_s, ev_y = st["scores"]["eval"], st["scores"]["labels"]
    print("\n[1] 較正 (合成データ: 真のロジットを %.1f 倍に潰した = 自信不足)"
          % st["true_temperature"])
    print(f"    N(fit)=2000  N(eval)={len(ev_s)}  正例率={float(np.mean(ev_y)):.3f}")
    print(f"    {'method':<12}{'ECE':>9}{'MCE':>9}{'Brier':>9}{'NLL':>9}   param")
    print(f"    {'(raw)':<12}{st['ece_before']:>9.4f}"
          f"{maximum_calibration_error(ev_s, ev_y, 15):>9.4f}"
          f"{st['brier_before']:>9.4f}{'-':>9}   -")
    ok_ece = True
    for m in ("temperature", "isotonic"):
        r = st[m]
        cal: SpanCalibrator = r["cal"]
        param = (f"T={cal.temperature_:.4f} (真値 {st['true_temperature']:.2f})"
                 if m == "temperature" else f"knots={len(cal.knots_x_ or [])}")
        print(f"    {m:<12}{r['ece']:>9.4f}"
              f"{maximum_calibration_error(r['probs'], ev_y, 15):>9.4f}"
              f"{r['brier']:>9.4f}{r['nll']:>9.4f}   {param}")
        ok_ece &= r["ece"] < st["ece_before"]
    print(f"    -> ECE は較正後に低下したか: {'OK' if ok_ece else 'FAIL'}")
    assert ok_ece, "calibration did not reduce ECE"

    # 保存・復元
    with tempfile.TemporaryDirectory() as td:
        paths = {}
        for m in ("temperature", "isotonic"):
            p = os.path.join(td, f"cal_{m}.json")
            st[m]["cal"].save(p)
            back = SpanCalibrator.load(p)
            a = st[m]["cal"].transform(ev_s[:200])
            b = back.transform(ev_s[:200])
            same = max(abs(x - y) for x, y in zip(a, b)) < 1e-12
            paths[m] = (os.path.getsize(p), same)
            assert same, f"{m}: save/load round-trip mismatch"
        print("    -> JSON 保存/復元一致: "
              + ", ".join(f"{m}={'OK' if v[1] else 'FAIL'} ({v[0]}B)" for m, v in paths.items()))

    # スパンごと較正
    demo_spans = [Span(0, 4, PIIType.NAME, "甲野花子"[:4], score=0.80),
                  Span(10, 14, PIIType.NAME, "乙川次郎"[:4], score=0.30)]
    tcal: SpanCalibrator = st["temperature"]["cal"]
    print("    -> transform_spans: "
          + ", ".join(f"{s.score:.2f}->{c.score:.3f}"
                      for s, c in zip(demo_spans, tcal.transform_spans(demo_spans))))

    # ---------------------------------------------------------- 2. 図 (フォント)
    font = use_japanese_font()
    print(f"\n[2] 図: 日本語フォント = {font or 'なし -> 英語ラベルに縮退'}")
    out_png = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "figures", "_selftest_reliability.png")
    fig = reliability_diagram(ev_s, ev_y, bins=15,
                              title="" if font else "Reliability (raw scores)",
                              out_path=out_png)
    print(f"    信頼性ダイアグラムを書き出し: {out_png} "
          f"({os.path.getsize(out_png)} bytes)")
    try:
        fig.canvas.get_renderer  # noqa: B018
    finally:
        _pyplot().close(fig)

    # ------------------------------------------------------------ 3. 突き合わせ
    toy = _selftest_metrics()
    g = toy["pos_docs"][0].sorted_spans()
    p_exact = [Span(g[0].start, g[0].end, g[0].label, g[0].text, score=0.9)]
    p_shift = [Span(g[0].start, g[0].end - 1, g[0].label, g[0].text[:-1], score=0.9)]
    tp_e, fp_e, fn_e = match_spans(g, p_exact, mode="exact")
    tp_s, fp_s, fn_s = match_spans(g, p_shift, mode="exact")
    tp_p, fp_p, fn_p = match_spans(g, p_shift, mode="partial")
    print("\n[3] match_spans (正解2件・予測1件)")
    print(f"    exact   / 完全一致 : tp={len(tp_e)} fp={len(fp_e)} fn={len(fn_e)}")
    print(f"    exact   / 1文字ずれ: tp={len(tp_s)} fp={len(fp_s)} fn={len(fn_s)}")
    print(f"    partial / 1文字ずれ: tp={len(tp_p)} fp={len(fp_p)} fn={len(fn_p)}"
          f"  (IoU={tp_p[0][0].iou(tp_p[0][1]):.3f})")
    assert (len(tp_e), len(tp_s), len(tp_p)) == (1, 0, 1)

    # 競合ケース: 1 つの正解に 2 つの予測が重なる -> 1 件だけ TP、残りは FP
    dup = [Span(g[0].start, g[0].end, g[0].label, g[0].text, score=0.6),
           Span(g[0].start, g[0].end - 2, g[0].label, g[0].text[:-2], score=0.9)]
    tp_d, fp_d, fn_d = match_spans(g, dup, mode="partial")
    print(f"    partial / 重複予測 : tp={len(tp_d)} fp={len(fp_d)} fn={len(fn_d)}"
          f"  (IoU最大の1件だけを対応づけ)")
    assert (len(tp_d), len(fp_d)) == (1, 1)

    # ---------------------------------------------------------- 4. 検出率/誤検出
    det = detection_rates(toy["pos_docs"], toy["pos_preds"], mode="partial")
    print("\n[4] detection_rates (全予測を採用: 正解6件を6件とも当てている)")
    print(f"    {'種別':<14}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
    for k, row in det["by_type"].items():
        print(f"    {row['ja']:<14}{row['precision']:>7.3f}{row['recall']:>7.3f}"
              f"{row['f1']:>7.3f}{row['support']:>9}")
    print(f"    micro P/R/F1 = {det['micro']['precision']:.3f} / "
          f"{det['micro']['recall']:.3f} / {det['micro']['f1']:.3f}"
          f"   macro R = {det['macro']['recall']:.3f}")
    assert abs(det["micro"]["recall"] - 1.0) < 1e-9

    rep = false_positive_report(toy["neg_docs"], toy["neg_preds"])
    print("\n[5] 誤検出率 3 定義 (否定例5文書・誤検出5件・全文書に1件ずつ)")
    print(f"    fp_per_doc         = {rep['fp_per_doc']:.3f} 件/文書   "
          f"({rep['n_fp']}/{rep['n_docs']})")
    print(f"    fp_per_1000_chars  = {rep['fp_per_1000_chars']:.3f} 件/1000字 "
          f"({rep['n_fp']}/{rep['n_chars']}字)")
    print(f"    doc_level_fp_rate  = {rep['doc_level_fp_rate']:.3f}  "
          f"<- README が「誤検出率」と呼ぶ定義 ({rep['n_docs_with_fp']}/{rep['n_docs']})")
    assert abs(false_positive_rate(toy["neg_docs"], toy["neg_preds"]) - 1.0) < 1e-9

    # ------------------------------------------------------- 6. 主要指標 (検算可)
    print("\n[6] recall_at_fixed_fpr (主要指標)")
    print("    陽性側スコア: 0.99 0.95 0.85 0.75 0.65 0.45 (各1件が正解1件に対応, 計6件)")
    print("    否定例スコア: 0.90 0.70 0.60 0.50 0.30 (5文書に1件ずつ)")
    print(f"    {'target':>8} {'threshold':>10} {'fpr':>7} {'recall':>8} {'手計算':>12}  note")
    # 手計算: target を満たす「最小の」閾値 = 検出率が最大になる閾値。
    #   0.00 -> 誤検出0件にするには 0.90 を落とす -> 閾値0.95 -> 拾える正解は 0.99,0.95 = 2/6
    #   0.20 -> 誤検出1件まで        -> 閾値0.75 -> 0.99,0.95,0.85,0.75      = 4/6
    #   0.60 -> 誤検出3件まで        -> 閾値0.60 -> さらに 0.65              = 5/6
    #   1.00 -> 誤検出5件まで        -> 候補最小の0.30 (否定例側の値) -> 全採用 = 6/6
    expect = {0.0: (0.95, 0.0, 2 / 6), 0.2: (0.75, 0.2, 4 / 6),
              0.6: (0.60, 0.6, 5 / 6), 1.0: (0.30, 1.0, 6 / 6)}
    for target in (0.0, 0.2, 0.6, 1.0):
        r = recall_at_fixed_fpr(toy["pos_docs"], toy["pos_preds"],
                                toy["neg_docs"], toy["neg_preds"],
                                target_fpr=target, mode="partial")
        et, ef, er = expect[target]
        ok = (abs(r["threshold"] - et) < 1e-9 and abs(r["fpr"] - ef) < 1e-9
              and abs(r["overall_recall"] - er) < 1e-9)
        print(f"    {target:>8.2f} {r['threshold']:>10.2f} {r['fpr']:>7.2f}"
              f" {r['overall_recall']:>8.3f} {er:>12.3f}  {'OK' if ok else 'MISMATCH'}"
              f" (curve {len(r['curve'])}点)")
        assert ok, f"recall_at_fixed_fpr mismatch at target={target}"

    r = recall_at_fixed_fpr(toy["pos_docs"], toy["pos_preds"],
                            toy["neg_docs"], toy["neg_preds"],
                            target_fpr=0.2, mode="partial")
    print("    種別別 (target=0.20):")
    for k, row in r["by_type"].items():
        print(f"      {row['ja']:<14} R={row['recall']:.3f} "
              f"(tp={row['tp']} fn={row['fn']} fp={row['fp']})")
    print(f"    curve 先頭3点 (threshold, fpr, recall): "
          + " ".join(f"({a:.2f},{b:.2f},{c:.2f})" for a, b, c in r["curve"][:3]))

    # 退化ケース
    print("\n[7] 退化ケース")
    r_nn = recall_at_fixed_fpr(toy["pos_docs"], toy["pos_preds"], [], [], target_fpr=0.05)
    print(f"    否定例0件      : threshold={r_nn['threshold']:.2f} "
          f"recall={r_nn['overall_recall']:.3f} degenerate={r_nn['degenerate']}")
    flat_pos = [[s.with_(score=0.5) for s in ps] for ps in toy["pos_preds"]]
    flat_neg = [[s.with_(score=0.5) for s in ps] for ps in toy["neg_preds"]]
    r_flat = recall_at_fixed_fpr(toy["pos_docs"], flat_pos, toy["neg_docs"], flat_neg,
                                 target_fpr=0.2)
    print(f"    全スコア同値   : threshold={r_flat['threshold']:.6f} "
          f"fpr={r_flat['fpr']:.2f} recall={r_flat['overall_recall']:.3f} "
          f"degenerate={r_flat['degenerate']}")
    r_none = recall_at_fixed_fpr(toy["pos_docs"], [[], [], []], toy["neg_docs"],
                                 [[] for _ in toy["neg_docs"]], target_fpr=0.05)
    print(f"    予測0件        : threshold={r_none['threshold']:.2f} "
          f"recall={r_none['overall_recall']:.3f} degenerate={r_none['degenerate']}")
    r_bad = recall_at_fixed_fpr(toy["pos_docs"], toy["pos_preds"], toy["neg_docs"],
                                toy["neg_preds"], target_fpr=-1.0)
    print(f"    達成不能(-1.0) : achieved={r_bad['achieved']} "
          f"fpr={r_bad['fpr']:.2f} degenerate={r_bad['degenerate']}")
    r_doc = recall_at_fixed_fpr(toy["pos_docs"], toy["pos_preds"], toy["neg_docs"],
                                toy["neg_preds"], target_fpr=0.2, fpr_metric="doc_level")
    print(f"    文書レベルFPR  : target=0.20 -> threshold={r_doc['threshold']:.2f} "
          f"fpr={r_doc['fpr']:.2f} recall={r_doc['overall_recall']:.3f}")
    assert r_none["overall_recall"] == 0.0 and not r_bad["achieved"]

    print("\n" + "=" * 78)
    print("すべての自己テストに合格 (較正でECE低下 / 突き合わせ / 3種の誤検出率 / 主要指標)")
    print("=" * 78)
