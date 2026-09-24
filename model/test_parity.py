#!/usr/bin/env python3
"""Check the MLX port against the torch reference (needs torch, transformers, the HF snapshots).

    python test_parity.py --mlx ~/Desktop/Yue2-3B-MLX

Compares AR logits, NAR velocity and VAE decode on tiny inputs. Exits non-zero on mismatch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from generate import CODEC_OFFSET, MUSIC_END, Tokenizer, token_prefix
from yue2_model import KVCache, load_model
from yue2_vae import load_vae

LATENT_START, LATENT_END, LATENT_PAD = 184621, 184622, 184623


def snapshot(repo):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, allow_patterns=["*.json", "*.py", "model.safetensors", "qwen.tiktoken"]))


def rel_err(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx", type=Path, default=Path.home() / "Desktop" / "Yue2-3B-MLX")
    parser.add_argument("--ref-model", default="m-a-p/YuE2-3B")
    parser.add_argument("--ref-vae", default="m-a-p/YuE2-Vae")
    args = parser.parse_args()
    from transformers import AutoModel, AutoModelForCausalLM

    ref_dir = snapshot(args.ref_model) if not Path(args.ref_model).is_dir() else Path(args.ref_model)
    vae_dir = snapshot(args.ref_vae) if not Path(args.ref_vae).is_dir() else Path(args.ref_vae)
    tok = Tokenizer(args.mlx / "qwen.tiktoken")
    prefix = token_prefix(tok, "pop, warm vocal", "[Verse]\nHello world\n", "off")
    codec = [7, 300, 5000, 12345, 32000, 9, 9, 42]
    ar_tokens = prefix + [c + CODEC_OFFSET for c in codec] + [MUSIC_END]
    frames = len(codec)
    rng = np.random.default_rng(0)
    state = rng.standard_normal((frames, 64)).astype(np.float32)
    raw_t = 0.3

    # ── torch reference ──
    ref = AutoModelForCausalLM.from_pretrained(ref_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                               local_files_only=True).eval()
    with torch.inference_mode():
        ref_logits = ref(torch.tensor([ar_tokens]), use_cache=False).logits[0].float().numpy()
        nar = [LATENT_START] + [LATENT_PAD] * frames + [LATENT_END]
        tokens = torch.tensor([ar_tokens + nar])
        ar_mask = torch.tensor([[True] * len(ar_tokens) + [False] * len(nar)])
        nar_mask = ~ar_mask
        content = torch.tensor([[False] * (len(ar_tokens) + 1) + [True] * frames + [False]])
        ref_v = ref.nar_velocity(tokens, ar_mask, nar_mask, content, torch.tensor(state), raw_t).float().numpy()
    del ref
    ref_vae = AutoModel.from_pretrained(vae_dir, trust_remote_code=True, decoder_only=True, local_files_only=True).eval()
    z = rng.standard_normal((1, 64, 40)).astype(np.float32)
    with torch.inference_mode():
        ref_audio = ref_vae.decode(torch.tensor(z))[0].T.numpy()  # [L,2]
        ref_tiled = ref_vae.decode_tiled(torch.tensor(z), core_frames=16, halo_frames=16)[0].T.numpy()
    del ref_vae

    # ── mlx ──
    model = load_model(args.mlx)
    caches = [KVCache() for _ in model.model.layers]
    mlx_logits = np.array(model.ar_step(mx.array([ar_tokens]), caches, all_positions=True)[0].astype(mx.float32))
    # incremental decode must equal the prefill logits for the same positions
    caches2 = [KVCache() for _ in model.model.layers]
    model.ar_step(mx.array([ar_tokens[:-3]]), caches2)
    step_logits = [np.array(model.ar_step(mx.array([[t]]), caches2).astype(mx.float32))[0] for t in ar_tokens[-3:]]
    ar_cache = model.nar_prefill(ar_tokens)
    mlx_v = np.array(model.nar_velocity(mx.array(state).astype(mx.bfloat16), raw_t, ar_cache, len(ar_tokens)).astype(mx.float32))
    vae = load_vae(args.mlx)
    zc = mx.array(z[0].T)  # [T,64]
    mlx_audio = np.array(vae(zc[None])[0])
    mlx_tiled = np.array(vae.decode_tiled(zc, core=16, halo=16))

    checks = {
        "ar_logits_rel_err": rel_err(mlx_logits, ref_logits),
        "ar_argmax_agree": float((mlx_logits.argmax(-1) == ref_logits.argmax(-1)).mean()),
        "ar_incremental_rel_err": rel_err(np.stack(step_logits), mlx_logits[-3:]),
        "nar_velocity_rel_err": rel_err(mlx_v, ref_v),
        "vae_rel_err": rel_err(mlx_audio, ref_audio),
        "vae_tiled_rel_err": rel_err(mlx_tiled, ref_tiled),
        "vae_len_ok": float(mlx_audio.shape == ref_audio.shape and mlx_tiled.shape == ref_tiled.shape),
    }
    for k, v in checks.items():
        print(f"{k:26s} {v:.5f}")
    ok = (checks["ar_logits_rel_err"] < 0.05 and checks["ar_argmax_agree"] > 0.9
          and checks["ar_incremental_rel_err"] < 0.05 and checks["nar_velocity_rel_err"] < 0.05
          and checks["vae_rel_err"] < 1e-3 and checks["vae_tiled_rel_err"] < 1e-3 and checks["vae_len_ok"])
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
