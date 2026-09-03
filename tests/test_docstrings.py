"""どの主張を実証するかが全 public API に明記されていることを機械的に検査する。

Claim: 検出率 / 低誤検出 / CPU速度 / 可逆性 — 「この関数はどの主張の証拠なのか」を
コード側に常駐させ、主張と実装が乖離しないようにする。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "sumi"

VALID_CLAIMS = {"検出率", "低誤検出", "CPU速度", "可逆性", "較正"}


def _public_defs(path: pathlib.Path):
    """モジュール内の public な関数・メソッド・クラスを列挙する。

    Claim: 検出率 — 検査対象を機械的に決め、見落としを無くす。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []

    def walk(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = child.name
                if name.startswith("_") and not name.startswith("__"):
                    continue
                if name in {"__post_init__", "__init__", "__repr__", "__eq__", "__hash__", "__str__"}:
                    continue
                qual = f"{prefix}{name}"
                out.append((qual, child))
                if isinstance(child, ast.ClassDef):
                    walk(child, prefix=f"{qual}.")

    walk(tree)
    return out


def _module_files():
    return sorted(p for p in PKG.rglob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_public_api_declares_a_claim(path: pathlib.Path):
    """全 public 定義の docstring に有効な ``Claim:`` 行があること。

    Claim: 検出率 / 低誤検出 / CPU速度 / 可逆性 — 契約 (docs/CONTRACT.md 規則3) の強制。
    """
    missing: list[str] = []
    bad: list[str] = []
    for qual, node in _public_defs(path):
        doc = ast.get_docstring(node)
        if not doc:
            missing.append(f"{qual} (docstring なし)")
            continue
        claim_lines = [ln for ln in doc.splitlines() if ln.strip().startswith("Claim:")]
        if not claim_lines:
            missing.append(f"{qual} (Claim: 行なし)")
            continue
        text = " ".join(claim_lines)
        if not any(c in text for c in VALID_CLAIMS):
            bad.append(f"{qual} -> {claim_lines[0].strip()!r}")

    rel = path.relative_to(ROOT)
    assert not missing, f"{rel}: Claim: 行が無い public 定義:\n  " + "\n  ".join(missing)
    assert not bad, (
        f"{rel}: Claim: が既定の主張語 {sorted(VALID_CLAIMS)} を含まない:\n  " + "\n  ".join(bad)
    )


def test_module_docstrings_present():
    """各モジュールがモジュール docstring を持つこと。

    Claim: 検出率 — モジュール単位でも「何の証拠か」を辿れるようにする。
    """
    bad = []
    for p in _module_files():
        tree = ast.parse(p.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, "モジュール docstring が無い: " + ", ".join(bad)
