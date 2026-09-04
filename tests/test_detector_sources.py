"""検出器がモデルをどこから読むか、読めなかったときにどう振る舞うかを検証する。

Claim: 検出率 — 「モデルを指定したのに、黙って規則層だけで動いていた」という
失敗は、利用者から見ると *検出漏れ* と区別がつかない。実際に公開直後の
モデルカードの使用例がこの状態だった (Hugging Face のリポジトリ ID が
``os.path.isdir`` で弾かれ、氏名を1件も検出できていなかった)。
その再発を止めるためのテスト。
"""

from __future__ import annotations

import os

import pytest

from sumi.detector import DEFAULT_MODEL_DIR, SumiDetector
from sumi.types import PIIType

HUB_ID = "NagaYu/sumi-ja-pii"

needs_model = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_MODEL_DIR),
    reason="no trained checkpoint (run scripts/train.py first)",
)


def test_bad_model_path_raises_instead_of_silently_degrading():
    """存在しないモデルを明示指定したら例外になること。

    Claim: 検出率 — 静かに規則層へ落ちると、利用者は「PIIが無かった」と
    誤解する。設定ミスは検出漏れに化けてはならない。
    """
    with pytest.raises(RuntimeError) as ei:
        SumiDetector("this/model-does-not-exist", device="cpu")
    msg = str(ei.value)
    assert "could not load" in msg
    assert "use_model=False" in msg, "規則層のみで動かす方法を案内していない"


def test_rules_only_is_still_possible_explicitly():
    """``use_model=False`` なら規則層のみで正常に動くこと。

    Claim: CPU速度 — モデル無しの軽量構成は正当な選択肢であり続ける。
    """
    det = SumiDetector(use_model=False)
    spans = det.detect("連絡先は090-1234-5678、メールは a@example.com です。")
    assert {s.label for s in spans} == {PIIType.PHONE, PIIType.EMAIL}
    assert det.info()["use_model"] is False


def test_missing_default_dir_falls_back_quietly(tmp_path, monkeypatch):
    """既定パスが無いだけなら例外にせず規則層で続行すること。

    Claim: 検出率 — 「明示指定の失敗」と「既定の探索が空振り」は区別する。
    未学習の状態でも import して使えることは、この設計の意図された性質。
    """
    monkeypatch.chdir(tmp_path)          # artifacts/sumi-model が無い場所へ
    det = SumiDetector()
    assert det.info()["use_model"] is False
    assert det.detect("電話は03-1234-5678です。")


@needs_model
def test_local_directory_loads_the_model():
    """ローカルディレクトリ指定でモデル層が有効になること。

    Claim: 検出率 — 学習成果物をそのまま読める経路の確認。
    """
    det = SumiDetector(DEFAULT_MODEL_DIR, device="cpu")
    assert det.info()["use_model"] is True
    spans = det.detect("田中太郎様の連絡先は090-1234-5678です。")
    assert any(s.label is PIIType.NAME and s.text == "田中太郎" for s in spans)


@pytest.mark.skipif(
    os.environ.get("SUMI_OFFLINE") == "1",
    reason="requires network access to the Hugging Face Hub",
)
@pytest.mark.parametrize("onnx", [False, True], ids=["pytorch", "onnx"])
def test_hub_repo_id_loads_the_model(onnx):
    """Hugging Face のリポジトリ ID を渡してもモデル層が有効になること。

    Claim: 検出率 — 公開モデルカードが案内している使い方そのもの。
    これが黙って規則層に落ちると、カードの例が氏名を検出できない。
    """
    try:
        det = SumiDetector(HUB_ID, device="cpu", onnx=onnx)
    except Exception as exc:  # ネットワーク不通は環境要因として扱う
        pytest.skip(f"Hub unreachable: {type(exc).__name__}: {exc}")
    assert det.info()["use_model"] is True, "Hub の ID がモデル層を有効にしていない"
    spans = det.detect("田中太郎様の連絡先は090-1234-5678、住所は東京都新宿区西新宿2-8-1です。")
    labels = {s.label for s in spans}
    assert PIIType.NAME in labels, "Hub 経由でモデルが効いていない (氏名が出ない)"
    assert PIIType.ADDRESS in labels
    assert PIIType.PHONE in labels
