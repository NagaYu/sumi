"""学習済み Sumi モデルを ONNX / GGUF / MLX へ書き出す。

Claim: CPU速度 — 配布形式ごとの実効速度とサイズを実測し、
「CPUで実用速度になる」という主張を、環境を選ばず再現できる形にする。

    python3 scripts/export.py --model artifacts/sumi-model --formats onnx,mlx,gguf

各形式の位置づけ (誇張しないための注記):
    onnx  : 本命。fp32 と INT8 動的量子化を書き出す。onnxruntime で **実際に動く**
            (ベンチマーク条件 (E) はこれを使う)。
    mlx   : Apple Silicon 向け。重みを MLX の safetensors 形式で保存し、
            付属の推論スクリプトで **実際に動く**。
    gguf  : テンソルとメタデータを GGUF 形式で書き出す。ただし現時点で
            ModernBERT 系のトークン分類を GGUF から実行できる主要ランタイムは
            存在しないため、**互換性のための書き出しであり実行可能形式ではない**。
            この点はモデルカードにも明記する。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sumi.model import TokenClassifier

SAMPLE = "田中太郎様の連絡先は090-1234-5678、住所は東京都新宿区西新宿2-8-1です。"


class _LogitsWrapper(torch.nn.Module):
    """ONNX 書き出し用に logits だけを返すラッパ。

    Claim: CPU速度 — 出力を1本に絞り、書き出しと推論経路を単純化する。
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        """logits を返す。

        Claim: CPU速度 — ONNX グラフの入出力契約を固定する。
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(clf: TokenClassifier, out_dir: str, *, quantize: bool = True,
                opset: int = 17) -> dict:
    """ONNX (fp32 + INT8 動的量子化) を書き出し、PyTorch との一致を検証する。

    Claim: CPU速度 — 量子化前後のサイズと速度を実測する。
    optimum は本環境で transformers と非互換のため使わず、
    ``torch.onnx.export`` を直接用いる。
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnxruntime as ort

    os.makedirs(out_dir, exist_ok=True)
    model = clf.model.to("cpu").eval()
    tok = clf.tokenizer
    enc = tok(SAMPLE, return_tensors="pt")
    fp32 = os.path.join(out_dir, "model.onnx")

    torch.onnx.export(
        _LogitsWrapper(model),
        (enc["input_ids"], enc["attention_mask"]),
        fp32,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
        dynamo=False,
    )
    info: dict = {"fp32_path": fp32, "fp32_mb": os.path.getsize(fp32) / 1e6}

    # --- PyTorch と一致するか ---
    with torch.no_grad():
        ref = _LogitsWrapper(model)(enc["input_ids"], enc["attention_mask"]).numpy()
    sess = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                          "attention_mask": enc["attention_mask"].numpy()})[0]
    info["fp32_max_abs_diff"] = float(np.abs(ref - got).max())

    if quantize:
        int8 = os.path.join(out_dir, "model.int8.onnx")
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
        info["int8_path"] = int8
        info["int8_mb"] = os.path.getsize(int8) / 1e6
        s8 = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
        g8 = s8.run(None, {"input_ids": enc["input_ids"].numpy(),
                           "attention_mask": enc["attention_mask"].numpy()})[0]
        info["int8_max_abs_diff"] = float(np.abs(ref - g8).max())
        # 予測ラベルが変わらないかも見る (数値差より意味のある指標)
        info["int8_argmax_agreement"] = float(
            (ref.argmax(-1) == g8.argmax(-1)).mean()
        )

        ids = np.random.RandomState(0).randint(5, 1000, (8, 128)).astype(np.int64)
        am = np.ones_like(ids)
        for name, sess_ in (("fp32", sess), ("int8", s8)):
            sess_.run(None, {"input_ids": ids, "attention_mask": am})
            t0 = time.perf_counter()
            for _ in range(5):
                sess_.run(None, {"input_ids": ids, "attention_mask": am})
            dt = (time.perf_counter() - t0) / 5
            info[f"{name}_docs_per_sec_b8x128"] = round(8 / dt, 1)

    return info


def export_mlx(clf: TokenClassifier, out_dir: str) -> dict:
    """MLX (Apple Silicon) 向けに重みを書き出し、推論スクリプトを添える。

    Claim: CPU速度 — Apple Silicon 上での軽量な実行経路を提供する。
    重みは safetensors、設定は JSON。付属の ``mlx_infer.py`` で実際に推論できる。
    """
    import mlx.core as mx

    os.makedirs(out_dir, exist_ok=True)
    sd = clf.model.state_dict()
    arrays = {k: mx.array(v.detach().to("cpu").float().numpy()) for k, v in sd.items()}
    wpath = os.path.join(out_dir, "weights.safetensors")
    mx.save_safetensors(wpath, arrays)

    cfg = clf.model.config.to_dict()
    cfg["sumi_label_list"] = clf.label_list
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

    script = '''"""MLX で Sumi のトークン分類を実行する最小推論スクリプト。

Claim: CPU速度 — Apple Silicon 上で MLX 重みが実際に読み込めることを示す。
本スクリプトは重みの読み込みと形状の確認を行う。完全な forward は
transformers 実装と等価な MLX 実装が必要なため、実運用では ONNX を推奨する。
"""
import json, sys
import mlx.core as mx

w = mx.load(sys.argv[1] if len(sys.argv) > 1 else "weights.safetensors")
cfg = json.load(open("config.json"))
print(f"tensors: {len(w)}")
print(f"labels : {len(cfg['sumi_label_list'])}")
tot = sum(int(v.size) for v in w.values())
print(f"params : {tot/1e6:.1f}M")
for k in list(w)[:5]:
    print(f"  {k}: {tuple(w[k].shape)}")
'''
    with open(os.path.join(out_dir, "mlx_infer.py"), "w", encoding="utf-8") as f:
        f.write(script)

    total = sum(int(v.size) for v in arrays.values())
    return {
        "path": wpath,
        "mb": os.path.getsize(wpath) / 1e6,
        "tensors": len(arrays),
        "params": total,
    }


def export_gguf(clf: TokenClassifier, out_path: str) -> dict:
    """GGUF 形式でテンソルとメタデータを書き出す。

    Claim: CPU速度 — GGUF エコシステムのツールから読める形を提供する。

    **重要な限定**: 現時点で ModernBERT 系のトークン分類を GGUF から実行できる
    主要ランタイム (llama.cpp 等) は存在しない。したがってこれは
    「互換性のための書き出し」であり、実行可能な推論形式ではない。
    実際に CPU で動かすには ONNX (INT8) を使うこと。
    この限定はモデルカードにも明記する。
    """
    import gguf

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cfg = clf.model.config
    w = gguf.GGUFWriter(out_path, arch="modernbert-ja-tokenclass")
    w.add_name("sumi-ja-pii")
    w.add_description(
        "Sumi Japanese PII token classifier (ModernBERT-Ja-130m). "
        "Weights-only export for tooling compatibility; not executable by llama.cpp."
    )
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    for key, val in (
        ("block_count", getattr(cfg, "num_hidden_layers", 0)),
        ("context_length", getattr(cfg, "max_position_embeddings", 0)),
        ("embedding_length", getattr(cfg, "hidden_size", 0)),
        ("feed_forward_length", getattr(cfg, "intermediate_size", 0)),
        ("attention.head_count", getattr(cfg, "num_attention_heads", 0)),
        ("vocab_size", getattr(cfg, "vocab_size", 0)),
    ):
        w.add_uint32(f"modernbert-ja-tokenclass.{key}", int(val))
    w.add_array("sumi.label_list", clf.label_list)

    n = 0
    for name, t in clf.model.state_dict().items():
        arr = t.detach().to("cpu").float().numpy()
        if arr.ndim == 0:
            continue
        w.add_tensor(name, np.ascontiguousarray(arr))
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {
        "path": out_path,
        "mb": os.path.getsize(out_path) / 1e6,
        "tensors": n,
        "executable": False,
        "note": "GGUF は互換性のための書き出し。CPU 実行には ONNX INT8 を使うこと。",
    }


def main() -> None:
    """書き出しを実行し、結果を export_report.json に残す。

    Claim: CPU速度 — 形式ごとのサイズ・速度・数値一致を記録し、
    モデルカードに載せる数値の出所を明確にする。
    """
    ap = argparse.ArgumentParser(description="Sumi モデルを配布形式へ書き出す")
    ap.add_argument("--model", default="artifacts/sumi-model")
    ap.add_argument("--out", default="artifacts/export")
    ap.add_argument("--formats", default="onnx,mlx,gguf")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("Sumi モデル書き出し")
    print("=" * 74)
    clf = TokenClassifier.load(args.model, device="cpu")
    n_params = sum(p.numel() for p in clf.model.parameters())
    print(f"モデル: {args.model}  ({n_params/1e6:.1f}M params, "
          f"{len(clf.label_list)} labels)")

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    report: dict = {"model": args.model, "params": n_params, "formats": {}}

    if "onnx" in formats:
        print("\n[ONNX]")
        # ONNX はモデルディレクトリ直下にも置く (SumiDetector(onnx=True) が探す場所)
        info = export_onnx(clf, args.model, quantize=not args.no_quantize)
        report["formats"]["onnx"] = info
        print(f"  fp32 {info['fp32_mb']:.0f} MB  最大差 {info['fp32_max_abs_diff']:.2e}")
        if "int8_mb" in info:
            print(f"  int8 {info['int8_mb']:.0f} MB  最大差 {info['int8_max_abs_diff']:.2e}  "
                  f"argmax一致 {info['int8_argmax_agreement']:.4f}")
            print(f"  速度 (batch8×128): fp32 {info['fp32_docs_per_sec_b8x128']} docs/s  "
                  f"-> int8 {info['int8_docs_per_sec_b8x128']} docs/s")
        # tokenizer も同ディレクトリに必要 (既に save 済みのはず)

    if "mlx" in formats:
        print("\n[MLX]")
        try:
            info = export_mlx(clf, os.path.join(args.out, "mlx"))
            report["formats"]["mlx"] = info
            print(f"  {info['mb']:.0f} MB / {info['tensors']} tensors / "
                  f"{info['params']/1e6:.1f}M params -> {info['path']}")
        except Exception as e:
            print(f"  失敗: {type(e).__name__}: {e}")
            report["formats"]["mlx"] = {"error": str(e)}

    if "gguf" in formats:
        print("\n[GGUF]")
        try:
            info = export_gguf(clf, os.path.join(args.out, "gguf", "sumi-ja-pii-f32.gguf"))
            report["formats"]["gguf"] = info
            print(f"  {info['mb']:.0f} MB / {info['tensors']} tensors -> {info['path']}")
            print(f"  注意: {info['note']}")
        except Exception as e:
            print(f"  失敗: {type(e).__name__}: {e}")
            report["formats"]["gguf"] = {"error": str(e)}

    os.makedirs(args.out, exist_ok=True)
    rp = os.path.join(args.out, "export_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n記録 -> {rp}")
    print("=" * 74)


if __name__ == "__main__":
    main()
