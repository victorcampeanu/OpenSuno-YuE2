# Models: download and use

OpenSuno ships no model weights. Everything is downloaded from Hugging Face at a pinned revision and
checked against the SHA-256 in [`model-assets.json`](../config/model-assets.json) (models) and
[`lora-assets.json`](../config/lora-assets.json) (LoRAs). No Hugging Face account or token is needed; all
repositories used are public.

## Packages

| Package | Needed for | Size | Source | Lands in |
|---|---|---|---|---|
| **YuE2 BF16** (`bf16`) | Generating songs on a Mac (MLX / Metal) | 7.5 GB | [ahmadw/YuE2-3B-MLX](https://huggingface.co/ahmadw/YuE2-3B-MLX) | `model/bf16/`, `model/8bit/` |
| **YuE2 CUDA BF16** (`cuda-bf16`) | Generating songs on Windows / NVIDIA | 7.8 GB | [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B), [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | `model/cuda/` |
| **YuE2 CUDA FP8** (`cuda-fp8`) | Same files as CUDA BF16, FP8 token generation | shared | same | `model/cuda/` |
| **Cover analysis** (`analysis`) | Covers and remixes of uploaded recordings | 2.8 GB | [m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2), [m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) | `transcriber/`, `mert/` |
| **Audio input** (`tokens`) | *Continue this recording*; needs Cover analysis | 0.3 GB | [Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4) (v9 files) | `tokens/` |

You need one generator (BF16 on a Mac, CUDA BF16 on Windows). The other packages are optional.
Leave about 20 GB free for weights plus the Python environments.

## Download from the page (recommended)

1. Install and start the Studio (see [README.md](../README.md#install)), then open
   <http://127.0.0.1:7862>.
2. Open **Models** and click **Download** next to each package you want.
3. Wait until the package shows as installed. Downloads run one file at a time, resume after an
   interruption or restart, and a file only replaces the old one after its SHA-256 matches.

When a song is created, the generator is picked in the model selector (**YuE2 BF16** by default on a
Mac, **YuE2 CUDA BF16** on Windows). After the download no internet connection is needed to generate.

### With a render node

If the Studio is paired with a render node (**Settings → Rendering**), the **Models** buttons download
on the node, not on the Studio computer. Directly against the node API:

```
curl -X POST -H "Authorization: Bearer $OPENSUNO_NODE_TOKEN" \
  "http://NODE-ADDRESS:7863/v1/models/download?model=bf16"
```

`model` accepts `bf16`, `cuda-bf16`, `cuda-fp8`, `analysis`, `tokens`, a LoRA id from
`config/lora-assets.json`, `loras` (every LoRA), `missing` (everything this machine can use) or `all`.
The same values work against the Studio at `POST http://127.0.0.1:7862/api/models/download?model=`.

## Download by hand

Useful for offline machines or a shared model folder. Install the Hugging Face CLI
(`pip install -U huggingface_hub`), run these from the OpenSuno folder, then restart the Studio. The
files are found by path and size, so they must end up exactly where shown.

```
# YuE2 BF16 (Mac)
hf download ahmadw/YuE2-3B-MLX bf16/model.safetensors 8bit/vae.safetensors 8bit/qwen.tiktoken \
  --revision fe0a9050fd658257b486b880422d8872ee1f81e3 --local-dir model

# YuE2 CUDA BF16 / FP8 (Windows)
hf download m-a-p/YuE2-3B model.safetensors qwen.tiktoken config.json generation_config.json \
  yue2_generation_config.json weights_manifest.json LICENSE THIRD_PARTY_NOTICES.md \
  licenses/SnakeBeta-NVIDIA-MIT.txt licenses/stable-audio-tools-MIT.txt \
  --revision 29b3558dd46954a0cd9021dc76d5c91864a0f1c7 --local-dir model/cuda/backbone
hf download m-a-p/YuE2-Vae model.safetensors config.json weights_manifest.json LICENSE \
  THIRD_PARTY_NOTICES.md licenses/SnakeBeta-NVIDIA-MIT.txt licenses/stable-audio-tools-MIT.txt \
  --revision 9a94e1d0ea9f8087e98f77fa88df4a4068104d2a --local-dir model/cuda/vae

# Cover analysis
hf download m-a-p/SheetSage2 model.safetensors \
  --revision eab522a8168e8b8b8c4856bf8609cd86198f01fe --local-dir transcriber
hf download m-a-p/MERT-v2-FullSong model.safetensors \
  --revision d8ba1c745e733b3908ce6ad16ebeb17ac7600a42 --local-dir mert

# Audio input
hf download Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 \
  tokenizer_head_joint_v9.safetensors nar_lora_joint_v9.safetensors \
  --revision e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5 --local-dir tokens
```

On a Mac the Studio links `model/bf16/vae.safetensors` and `qwen.tiktoken` to the `model/8bit/`
copies at startup. To check a file against the manifest: `shasum -a 256 <file>` (macOS) or
`certutil -hashfile <file> SHA256` (Windows).

## LoRAs

LoRAs are small adapters that push YuE2 toward a style (how the song is written) or a sound (how it
is rendered).

- **Models → LoRAs** lists the documented community adapters from
  [`lora-assets.json`](../config/lora-assets.json): YuE2 instrumental (Mothersuperior v3), Old School
  Hip-Hop, the MLTNT / CHNSN / QWWL / DRKSF / CNZN artist packs and the Real-audio decoder v9.
  Downloaded files go to `loras/`.
- Any other YuE2 LoRA in `.safetensors` (Hugging Face, PEFT, ComfyUI or Sound & Vision layout) can be
  dropped into `loras/` by hand. It appears in the picker the next time it opens. See
  [`loras/README.md`](../loras/README.md).

To use one, open **More Options** when creating a song:

- **Writes** (the style LoRA): choosing a documented adapter fills in its recommended strength, plan
  and style influence. Its trigger word (for example `mltnt` or `sv_oldschoolhiphop`) should start
  the style line.
- **Renders** (the sound LoRA): for sound-only adapters such as the Real-audio decoder v9. It stacks
  with the Writes LoRA.

On covers the writing half of a style LoRA is dropped after the arrangement step so the melody is
not rewritten into a loop; only its sound half, if it has one, stays. Details are in
[FEATURES.md](FEATURES.md).

## Licenses

All model weights above are **CC BY-NC 4.0 (non-commercial)** unless the model card says otherwise;
read each model card before using them, or music made with them, commercially. Each LoRA carries its
author's license on its Hugging Face page. OpenSuno's own code is MIT (see [LICENSE](../LICENSE)); that does not relicense the
models. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
