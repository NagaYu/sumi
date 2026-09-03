"""(C) ローカル4B級LLM にプロンプトで検出させる条件 — 現行の回避策。

Claim: 検出率 / CPU速度 — 「小型モデルでは日本語PIIが無理だから 4B 級LLMを回す」
という現行の回避策を、**同じCPU上で**同じ入力に対して実測し、
精度と速度の両面で Sumi と正面から比較する。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from sumi.types import PIIType, Source, Span
from benchmarks.baselines import BaselineInfo

DEFAULT_REPO = "lmstudio-community/Qwen3-4B-Instruct-2507-GGUF"
DEFAULT_FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

#: LLM に出させる日本語キー -> Sumi 種別
JA_KEY_TO_TYPE: dict[str, PIIType] = {
    "氏名": PIIType.NAME,
    "人名": PIIType.NAME,
    "名前": PIIType.NAME,
    "住所": PIIType.ADDRESS,
    "電話番号": PIIType.PHONE,
    "電話": PIIType.PHONE,
    "メールアドレス": PIIType.EMAIL,
    "メール": PIIType.EMAIL,
    "生年月日": PIIType.DOB,
    "金融口座": PIIType.BANK_ACCOUNT,
    "口座": PIIType.BANK_ACCOUNT,
    "口座番号": PIIType.BANK_ACCOUNT,
    "カード番号": PIIType.CREDIT_CARD,
    "クレジットカード": PIIType.CREDIT_CARD,
    "マイナンバー": PIIType.MYNUMBER,
    "個人番号": PIIType.MYNUMBER,
    "会員番号": PIIType.MEMBER_ID,
    "顧客番号": PIIType.MEMBER_ID,
    "郵便番号": PIIType.POSTAL_CODE,
}

PROMPT = """あなたは日本語の個人情報(PII)検出器です。
次のテキストから個人情報だけを抜き出し、JSON配列のみを出力してください。

各要素は {{"type": 種別, "text": 抜き出した文字列}} の形式です。
種別は次のいずれか: 氏名, 住所, 電話番号, メールアドレス, 生年月日, 金融口座, カード番号, マイナンバー, 会員番号, 郵便番号

重要な注意:
- 個人情報でないものを含めてはいけません。
- 普通名詞・地名・企業名・施設名・型番・注文番号・議案番号・契約日は個人情報ではありません。
- "text" はテキスト中に現れる通りに、一字一句そのまま書き写してください。
- 敬称(様/さん/氏/殿)は含めないでください。
- 該当が無ければ [] とだけ出力してください。
- JSON配列以外は一切出力しないでください。

テキスト:
{text}
"""


@dataclass
class LLMConfig:
    """(C) の実行設定。

    Claim: CPU速度 — GPUオフロードを 0 に固定し、CPU のみの実測値であることを保証する。
    """

    repo: str = DEFAULT_REPO
    filename: str = DEFAULT_FILE
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0          # CPU 比較のため必ず 0
    max_tokens: int = 512
    temperature: float = 0.0


class LocalLLMBaseline:
    """llama.cpp で 4B 級 LLM を CPU 実行し、プロンプトで PII を抽出させる条件。

    Claim: 検出率 / CPU速度 — 精度が出たとしても CPU では桁違いに遅い、
    という現行回避策のトレードオフを定量化する。
    """

    name = "local_llm_4b"
    label = "(C) ローカルLLM 4B (Q4)"

    def __init__(self, cfg: LLMConfig | None = None, model_path: str | None = None) -> None:
        self.cfg = cfg or LLMConfig()
        self.model_path = model_path
        self._llm = None

    def info(self) -> BaselineInfo:
        """条件のメタ情報 (散布図の座標)。

        Claim: CPU速度 — 4.0e9 パラメータ・4bit量子化という規模を明示し、
        0.13e9 の Sumi との対比を可能にする。
        """
        return BaselineInfo(
            name=self.name,
            label=self.label,
            params=4.0e9,
            runtime="llama.cpp (CPU, n_gpu_layers=0)",
            quantization="Q4_K_M",
            notes="Qwen3-4B-Instruct-2507 をプロンプトで PII 抽出に用いる",
        )

    def resolve_path(self) -> str | None:
        """GGUF のローカルパスを解決する (無ければ取得を試みる)。

        Claim: CPU速度 — モデル取得は計測対象外。事前に解決しておく。
        """
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(self.cfg.repo, self.cfg.filename)
            self.model_path = p
            return p
        except Exception:
            return None

    def available(self) -> bool:
        """llama_cpp と GGUF が揃っているか。

        Claim: 検出率 — 環境不備を検出漏れと取り違えないため。
        """
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            return False
        return self.resolve_path() is not None

    def warmup(self) -> None:
        """モデルをロードし、1回だけ短い生成を回す。

        Claim: CPU速度 — 2.5GB のロードと最初のグラフ構築を計測から除き、
        定常状態のスループットだけを比較する。
        """
        if self._llm is not None:
            return
        from llama_cpp import Llama

        path = self.resolve_path()
        if path is None:
            raise RuntimeError("GGUF model not available")
        self._llm = Llama(
            model_path=path,
            n_ctx=self.cfg.n_ctx,
            n_threads=self.cfg.n_threads,
            n_gpu_layers=self.cfg.n_gpu_layers,   # 0 = CPU only
            verbose=False,
        )
        self._llm.create_chat_completion(
            [{"role": "user", "content": "こんにちは"}], max_tokens=8, temperature=0.0
        )

    def detect(self, text: str) -> list[Span]:
        """LLM に PII を列挙させ、原文中の位置へ写像して Span にする。

        Claim: 検出率 — LLM は「文字列」しか返さないため、原文照合で
        オフセットを復元する。復元できなかった抽出は **検出とみなさない**
        (幻覚した文字列を正解扱いしないため)。
        """
        self.warmup()
        assert self._llm is not None
        out = self._llm.create_chat_completion(
            [{"role": "user", "content": PROMPT.format(text=text)}],
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        content = out["choices"][0]["message"]["content"] or ""
        items = _parse_json_items(content)
        return _align_to_text(items, text, self.name)


def _parse_json_items(content: str) -> list[tuple[str, str]]:
    """LLM 出力から (種別, 文字列) の組を頑健に取り出す。

    Claim: 検出率 — JSON が壊れていても取れるだけ取ることで、
    整形能力の低さを検出能力の低さとして不当に計上しない
    (基準線に有利側の処理)。
    """
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    # <think> ブロックを除去
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)

    items: list[tuple[str, str]] = []
    m = re.search(r"\[.*\]", content, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            for el in data:
                if isinstance(el, dict):
                    t = str(el.get("type") or el.get("種別") or "").strip()
                    v = el.get("text") or el.get("値") or el.get("value")
                    if isinstance(v, list):
                        for vv in v:
                            items.append((t, str(vv)))
                    elif v is not None:
                        items.append((t, str(v)))
                    # {"氏名": "田中太郎"} 形式にも対応
                    if not t and not v:
                        for k, vv in el.items():
                            if isinstance(vv, str):
                                items.append((str(k), vv))
            if items:
                return items
        except Exception:
            pass
    # JSON が壊れている場合の正規表現フォールバック
    for mm in re.finditer(r'"(?:type|種別)"\s*:\s*"([^"]+)"\s*,\s*"(?:text|値|value)"\s*:\s*"([^"]+)"', content):
        items.append((mm.group(1), mm.group(2)))
    return items


def _align_to_text(items: list[tuple[str, str]], text: str, baseline_name: str) -> list[Span]:
    """抽出文字列を原文に照合してオフセットを与える。

    Claim: 検出率 / 低誤検出 — 同じ文字列が複数回出る場合は未使用の出現を
    順に割り当て、原文に存在しない文字列 (幻覚) は捨てる。
    """
    used: list[tuple[int, int]] = []
    spans: list[Span] = []
    for raw_type, value in items:
        t = None
        for k, v in JA_KEY_TO_TYPE.items():
            if k in raw_type:
                t = v
                break
        if t is None or not value:
            continue
        value = value.strip()
        if not value or value not in text:
            continue  # 幻覚 or 表記改変 -> 検出とみなさない
        start = -1
        pos = 0
        while True:
            i = text.find(value, pos)
            if i < 0:
                break
            if all(not (i < b and a < i + len(value)) for a, b in used):
                start = i
                break
            pos = i + 1
        if start < 0:
            continue
        end = start + len(value)
        used.append((start, end))
        spans.append(
            Span(
                start=start, end=end, label=t, text=value,
                score=0.9, source=Source.BASELINE,
                meta={"baseline": baseline_name, "llm_type": raw_type},
            )
        )
    return sorted(spans, key=lambda s: (s.start, s.end))


if __name__ == "__main__":
    import time

    b = LocalLLMBaseline()
    print("available:", b.available())
    if b.available():
        t0 = time.perf_counter()
        b.warmup()
        print(f"load+warmup: {time.perf_counter()-t0:.1f}s")
        tests = [
            "田中太郎様よりご連絡をいただきました。連絡先は090-1234-5678、tanaka.taro@example.co.jp です。住所は東京都新宿区西新宿2-8-1、生年月日は1985年3月4日。",
            "森の中を歩いていると、林業の振興について泉が湧くように話が広がった。",
            "型番TX-2024-0355、注文番号0120-8834-221でお問い合わせください。契約日は2023年4月1日。",
        ]
        for t in tests:
            t0 = time.perf_counter()
            sp = b.detect(t)
            print(f"\n[{time.perf_counter()-t0:.2f}s] {t[:44]}")
            for s in sp:
                print(f"   {s.label.ja:8s} {s.slice_of(t)!r}")
            if not sp:
                print("   (検出なし)")
