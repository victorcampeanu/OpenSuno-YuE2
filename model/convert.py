#!/usr/bin/env python3
"""Convert m-a-p/YuE2-3B + m-a-p/YuE2-Vae into an MLX package directory.

Output layout (HF-uploadable):
  model.safetensors  bf16 backbone, or affine-quantized Linear layers (--quantize)
  vae.safetensors    f32 Oobleck decoder, weight norm folded, MLX conv layout
  config.json, vae_config.json, qwen.tiktoken, yue2_generation_config.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from yue2_model import Yue2Model

TIME_EMBEDDER = {"time_embedder.mlp.0.": "time_embedder.fc1.", "time_embedder.mlp.2.": "time_embedder.fc2."}
CONV_TRANSPOSE = re.compile(r"^layers\.\d+\.layers\.1\.weight_v$")


def resolve(repo_or_dir: str, patterns: list[str]) -> Path:
    if Path(repo_or_dir).is_dir():
        return Path(repo_or_dir)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_or_dir, allow_patterns=patterns))


def rename(key: str) -> str:
    for old, new in TIME_EMBEDDER.items():
        if key.startswith(old):
            return new + key[len(old):]
    return key


def fold_weight_norm(weights: dict) -> dict:
    """decoder.* (g, v) pairs → single MLX-layout conv weights."""
    out = {}
    for key, value in weights.items():
        if not key.startswith("decoder.") or key.endswith("weight_g"):
            continue
        key = key[len("decoder."):]
        if key.endswith("weight_v"):
            g = weights["decoder." + key[:-1] + "g"]
            norm = mx.sqrt(mx.sum(value.astype(mx.float32) ** 2, axis=(1, 2), keepdims=True))
            w = g * value / norm
            # torch Conv1d [out,in,k] → mlx [out,k,in]; ConvTranspose1d [in,out,k] → mlx [out,k,in]
            w = w.transpose(1, 2, 0) if CONV_TRANSPOSE.match(key) else w.transpose(0, 2, 1)
            out[key[:-2]] = w
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="m-a-p/YuE2-3B", help="HF repo id or local snapshot dir")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae", help="HF repo id or local snapshot dir")
    parser.add_argument("--output", type=Path, default=Path.home() / "Desktop" / "Yue2-3B-MLX")
    parser.add_argument("-q", "--quantize", action="store_true", help="affine-quantize backbone Linear layers")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--nar-bits", type=int, choices=[4, 8], help="bits for the NAR path (nar_self_attn/nar_mlp); default 8 (flow matching degrades at 4 bits)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    model_dir = resolve(args.model, ["model.safetensors", "config.json", "qwen.tiktoken", "yue2_generation_config.json"])
    vae_dir = resolve(args.vae, ["model.safetensors", "config.json"])

    cfg = json.loads((model_dir / "config.json").read_text())
    weights = {rename(k): v for k, v in mx.load(str(model_dir / "model.safetensors")).items()}
    if args.quantize:
        model = Yue2Model(cfg)
        model.load_weights(list(weights.items()))
        nar_bits = args.nar_bits or 8

        def predicate(path, module):
            if not isinstance(module, nn.Linear) or not (path.startswith("model.layers") or path == "lm_head"):
                return False
            return {"group_size": args.group_size, "bits": nar_bits if ".nar_" in path else args.bits}

        nn.quantize(model, args.group_size, args.bits, class_predicate=predicate)
        weights = dict(tree_flatten(model.parameters()))
        cfg["quantization"] = {"group_size": args.group_size, "bits": args.bits, "nar_bits": nar_bits, "mode": "affine"}
    print(f"[write] model.safetensors ({len(weights)} tensors)", flush=True)
    mx.save_safetensors(str(args.output / "model.safetensors"), weights, metadata={"format": "mlx"})
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    vae = fold_weight_norm(mx.load(str(vae_dir / "model.safetensors")))
    print(f"[write] vae.safetensors ({len(vae)} tensors)", flush=True)
    mx.save_safetensors(str(args.output / "vae.safetensors"), vae, metadata={"format": "mlx"})
    shutil.copy2(vae_dir / "config.json", args.output / "vae_config.json")
    for name in ("qwen.tiktoken", "yue2_generation_config.json"):
        shutil.copy2(model_dir / name, args.output / name)
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
