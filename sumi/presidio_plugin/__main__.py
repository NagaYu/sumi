"""``python3 -m sumi.presidio_plugin`` で自己テストを実行するためのエントリ。

Claim: 検出率 — プラグインの動作確認を他モジュールと同じ作法で行えるようにする。
"""

from sumi.presidio_plugin import _selftest

if __name__ == "__main__":
    _selftest()
