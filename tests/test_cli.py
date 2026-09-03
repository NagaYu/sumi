"""CLI の契約 (redact / restore の往復) を検証する。

Claim: 可逆性 — 仕様に明記されたコマンド
``sumi redact input.txt --out masked.txt --map map.json`` が
実際に動き、対応表を使って原文が完全に戻ることを確認する。
"""

from __future__ import annotations

import json
import os

import pytest

from sumi.cli import main
from sumi.detector import DEFAULT_MODEL_DIR
from sumi.synth import build_documents

#: 学習済みモデルが無い環境 (CI など) では、モデル層に依存する検査を飛ばす
needs_model = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_MODEL_DIR),
    reason="no trained checkpoint (run scripts/train.py first)",
)

SAMPLE = """ご担当 田中太郎 様

平素より大変お世話になっております。
ご連絡先は 090-1234-5678、メールは taro.tanaka@example.co.jp です。
ご住所は 〒160-0023 東京都新宿区西新宿2-8-1 とのことでした。
なお型番 TX-2024-0355 は生産終了、契約日は2023年4月1日です。
"""


def _run(args) -> int:
    return main(args)


def test_redact_restore_roundtrip(tmp_path):
    """redact -> restore で原文が1バイトも違わず戻ること。

    Claim: 可逆性 — CLI 経路での主張(2)。
    """
    src = tmp_path / "input.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    masked = tmp_path / "masked.txt"
    mp = tmp_path / "map.json"

    assert _run(["redact", str(src), "--out", str(masked), "--map", str(mp), "-q"]) == 0
    assert masked.exists() and mp.exists()

    out = tmp_path / "restored.txt"
    assert _run(["restore", str(masked), "--map", str(mp), "--out", str(out), "-q"]) == 0
    assert out.read_text(encoding="utf-8") == SAMPLE


def test_map_file_is_owner_only(tmp_path):
    """CLI が保存する対応表が 0600 であること。

    Claim: 可逆性 — 復元の鍵をファイル権限でも守る。
    """
    src = tmp_path / "input.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    mp = tmp_path / "map.json"
    _run(["redact", str(src), "--out", str(tmp_path / "m.txt"), "--map", str(mp), "-q"])
    assert (os.stat(mp).st_mode & 0o777) == 0o600


def test_masked_output_contains_no_originals(tmp_path):
    """マスク済み出力に元値が残っていないこと。

    Claim: 可逆性 — CLI 経路での主張(1)に相当する検査。
    外部へ渡るのはこのファイルなので、ここに元値があってはならない。
    """
    src = tmp_path / "input.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    masked = tmp_path / "masked.txt"
    mp = tmp_path / "map.json"
    _run(["redact", str(src), "--out", str(masked), "--map", str(mp), "-q"])

    body = masked.read_text(encoding="utf-8")
    mapping = json.loads(mp.read_text(encoding="utf-8"))
    for e in mapping["entries"]:
        assert e["original"] not in body, f"マスク済み出力に元値が残っている: {e['original']}"
    assert mapping["entries"], "何も検出されていない (テストが無意味になる)"


def test_detect_json_output_is_valid(tmp_path, capsys):
    """``sumi detect --json`` が妥当な JSON を出すこと。

    Claim: 検出率 — 機械可読な出力形式を保証する。
    """
    src = tmp_path / "input.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    assert _run(["detect", str(src), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "spans" in payload and "timings" in payload
    for s in payload["spans"]:
        assert {"start", "end", "label", "score"} <= set(s)
        assert payload["text"][s["start"]:s["end"]] == s["text"]


def test_roundtrip_on_synthetic_documents(tmp_path):
    """合成文書でも CLI 往復が成立すること。

    Claim: 可逆性 — 1つの手書き例に依存しないことを確認する。
    """
    for i, d in enumerate(build_documents(8, seed=606)):
        src = tmp_path / f"in{i}.txt"
        src.write_text(d.text, encoding="utf-8")
        masked = tmp_path / f"m{i}.txt"
        mp = tmp_path / f"map{i}.json"
        out = tmp_path / f"out{i}.txt"
        _run(["redact", str(src), "--out", str(masked), "--map", str(mp), "-q"])
        _run(["restore", str(masked), "--map", str(mp), "--out", str(out), "-q"])
        assert out.read_text(encoding="utf-8") == d.text


@needs_model
def test_no_rules_flag_disables_rule_layer(tmp_path, capsys):
    """``--no-rules`` で規則層が無効になること。

    Claim: 低誤検出 — 利用者が層を選べる設計 (用途に応じた調整) を保証する。

    件数では判定しない。モデルが十分に学習されると、規則層が無くても
    同じ件数を検出できてしまうため (実際にそうなった)。
    代わりに **スパンの出所** を見る: 規則層を切れば ``from == "rule"`` の
    スパンは1件も出てはならない。
    """
    src = tmp_path / "input.txt"
    src.write_text(SAMPLE, encoding="utf-8")

    _run(["detect", str(src), "--json"])
    with_rules = json.loads(capsys.readouterr().out)
    _run(["detect", str(src), "--json", "--no-rules"])
    without = json.loads(capsys.readouterr().out)

    src_of = lambda payload: {s["meta"].get("from") for s in payload["spans"]}
    assert "rule" in src_of(with_rules), "規則層が有効なのに rule 由来のスパンが無い"
    assert "rule" not in src_of(without), "--no-rules なのに rule 由来のスパンが出ている"
    assert without["spans"], "--no-rules で何も検出されないのは想定外"


def test_no_model_flag_disables_model_layer(tmp_path, capsys):
    """``--no-model`` でモデル層が無効になること。

    Claim: CPU速度 — 規則層だけの軽量構成を選べることを保証する。
    """
    src = tmp_path / "input.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    _run(["detect", str(src), "--json", "--no-model"])
    payload = json.loads(capsys.readouterr().out)
    assert all(s["meta"].get("from") == "rule" for s in payload["spans"])
    assert payload["timings"]["model"] < 0.01, "モデル層が動いている"
