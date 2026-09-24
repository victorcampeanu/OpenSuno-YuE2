# Third-party notices

OpenSuno's own code is MIT-licensed (see [LICENSE](LICENSE)). The files below come from, or are
adapted from, other projects and stay under their original licenses. Where a folder is listed, the
whole folder is covered unless a file says otherwise.

| Files | Source | License |
|---|---|---|
| `transcriber/` | [m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2), revision `eab522a8168e8b8b8c4856bf8609cd86198f01fe` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `mert/` | [m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong), revision `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `model/README.md`, `model/convert.py`, `model/generate.py`, `model/test_parity.py`, `model/yue2_model.py`, `model/yue2_vae.py`, `model/8bit/*.json`, `model/bf16/*.json` | [ahmadw/YuE2-3B-MLX](https://huggingface.co/ahmadw/YuE2-3B-MLX), revision `fe0a9050fd658257b486b880422d8872ee1f81e3`, a port of [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); the SnakeBeta activation (NVIDIA) and stable-audio-tools decoder code (Stability AI) inside are MIT |
| `vendor/yue2_abc.py` | [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) | Apache-2.0, see [`vendor/YuE-LICENSE`](vendor/YuE-LICENSE) |
| `web/style-presets*.js`, `web/lyrics-structure.js` (adapted) | [TheLocalLab/ComfyUI-SongScribe](https://github.com/TheLocalLab/ComfyUI-SongScribe) | MIT, Copyright (c) 2026 TheLocalLab |
| `web/icons/` | [Heroicons](https://github.com/tailwindlabs/heroicons) | MIT, Copyright (c) Tailwind Labs, Inc., see [`web/icons/LICENSE`](web/icons/LICENSE) |

The CC BY-NC 4.0 files above may be shared and adapted for non-commercial purposes with
attribution. Anyone who wants to use OpenSuno commercially has to replace or separately license
them.

## Model weights

No weights are stored in this repository. They are downloaded from their Hugging Face repositories
at pinned revisions (see [MODELS.md](MODELS.md)) and are licensed by their authors: YuE2-3B,
YuE2-Vae, SheetSage2, MERT-v2 and the Mothersuperior real-audio tokenizer are CC BY-NC 4.0. Each
community LoRA in [`lora-assets.json`](lora-assets.json) is licensed as stated on its model page.

OpenSuno is an independent project and is not affiliated with or endorsed by Suno, Inc.
