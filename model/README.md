---
license: cc-by-nc-4.0
base_model: m-a-p/YuE2-3B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-to-audio
tags:
- mlx
- music-generation
- yue2
language:
- en
- zh
---

# YuE2-3B MLX

Native [MLX](https://github.com/ml-explore/mlx) port of [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) (AR/NAR
Mixture-of-Transformers music generation) with the [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) decoder folded in.
Runs on Apple Silicon with no PyTorch dependency. Lyrics + style prompt in, 48 kHz stereo song out, with optional
editable ABC score planning.

## Variants

| Folder | Backbone | Size | AR logits / NAR velocity error vs torch bf16 |
|---|---|---|---|
| `bf16/` | bf16 | 7.0 GB | 1.4% / 1.6% |
| `8bit/` | 8-bit affine, group size 64, all backbone Linear layers | 4.2 GB | 1.0% / 1.9% |
| `4bit/` | 4-bit AR path, 8-bit NAR path, group size 64 | 3.4 GB | 8.1% / 4.4% |

The NAR (flow-matching) path stays at 8 bits in the 4-bit variant because fully 4-bit flow matching drifts
(12% velocity error). VAE decode is bit-exact in all variants. The Studio downloads and offers `bf16/` only; the
quantized variants are for `convert.py` users running the CLI, where `8bit/` gives near-bf16 output at roughly twice
the decode speed.

Each folder contains `model.safetensors`, `vae.safetensors`, `config.json`, `vae_config.json`, `qwen.tiktoken`
and `yue2_generation_config.json`. The Python files at the repo root are the inference code.

## Usage

```bash
pip install mlx tiktoken numpy huggingface_hub
hf download ahmadw/YuE2-3B-MLX --include "*.py" "8bit/*" --local-dir YuE2-3B-MLX
cd YuE2-3B-MLX
python generate.py --model 8bit \
  --style "English, indie pop, bright acoustic guitar, soft drums, warm lead vocal" \
  --lyrics "[Verse]
Soft morning light is touching the window.
[Chorus]
Stay with the rhythm, let it carry us home." \
  --cot full --seed 831001 --out song.wav
```

Options: `--cot off|melody|full` (symbolic ABC planning; `full` is the upstream default), `--abc-file` to supply your
own score, `--cfg-scale`, `--steps` (NAR midpoint steps, default 32), `--max-semantic-tokens` (song length cap, 25
frames per second), `--decode-latents <out>.latents.npy` to re-decode a saved run. A generated ABC score is written
next to the wav.

Measured on an M-series Mac: bf16 decodes at about 70 tokens/s (35 with CFG), 8bit at about 80 with CFG.
A three-minute song takes a few minutes end to end.

## Converting yourself

```bash
python convert.py --output YuE2-3B-MLX/8bit -q --bits 8          # or --bits 4 (NAR stays 8-bit), or no -q for bf16
python test_parity.py --mlx YuE2-3B-MLX/8bit                      # needs torch + transformers
```

Sampling uses MLX's RNG, so seeds reproduce within MLX only.

## License

Weights are derived from m-a-p/YuE2-3B and YuE2-Vae and inherit their **CC BY-NC 4.0** license (non-commercial).
The SnakeBeta activation and stable-audio-tools decoder code are MIT-licensed by NVIDIA and Stability AI respectively;
see the upstream `THIRD_PARTY_NOTICES.md`. Please cite the original YuE2 authors (m-a-p) when using these weights.
