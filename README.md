# OpenSuno

Local music generation with a browser interface for YuE2 on Apple Silicon and Windows/NVIDIA:
lyrics and style prompts, instrumental mode, covers of uploaded recordings, song editing and
extension, community LoRAs, a library with projects, and an optional render node on another
machine. An independent interface; not affiliated with Suno.

- [START-HERE.md](START-HERE.md): how to use the page and what each control does.
- [FEATURES.md](FEATURES.md): every feature, how it works and what it improves.
- [MODELS.md](MODELS.md): which models to download, how (from the page or by hand) and how to use
  LoRAs.

## Install

Allow at least 20 GB of free disk space for weights and Python environments. Tested on an M4 Pro
with 64 GB and on an RTX 4070 Ti SUPER 16 GB; other hardware has not been benchmarked here.

### Mac: disk image

`OpenSuno.dmg` is one installer for a Mac with nothing else installed (no Homebrew, no Terminal).
Open it and open **Install OpenSuno**. Tick what this Mac should do:

- **Studio**: the interface. It opens in the browser and sends everything heavy to a render node
  (rendering, cover analysis, trimming uploads, and the OpenAI calls for styles and album art; the
  OpenAI key is kept on the node).
- **Render node**: renders songs on this Mac's GPU, with the **OpenSuno Render Node** menu bar app.
  Optionally with **Cover analysis** (the PyTorch environment for turning recordings into covers)
  and **Start at login**.

Both on one Mac is the all-in-one setup. The installer puts the code, a self-contained Python and
ffmpeg into `~/OpenSuno`, downloads the Python packages the chosen parts need (internet required),
installs the apps into /Applications and starts them. Running it again updates in place without
touching songs, settings or models. The installer is not notarized: on the first open go to
**System Settings → Privacy & Security** and click **Open Anyway**.

A Studio on another Mac pairs in **Settings → Rendering** with the node's address and token (both
under the menu bar icon). To build the disk image on a Mac with Xcode:
`Build OpenSuno Installer.command` writes `dist/OpenSuno.dmg`.

### Mac: from source

1. Clone the repository and install [Homebrew](https://brew.sh) if necessary.
2. `bash "Install OpenSuno.command"` installs Python 3.12 and 3.11, FFmpeg and two isolated Python
   environments (MLX for generation, PyTorch for cover analysis). It can be rerun after a failure.
3. Open `Launch Studio.command`. The page at <http://127.0.0.1:7862> lists the model packages under
   **Models**: **BF16** (the generator, about 7.5 GB), **Cover analysis** (SheetSage2 and MERT-v2,
   about 2.8 GB, for uploaded recordings) and **Audio input** (about 0.3 GB, needs Cover analysis,
   for *Continue this recording*).

Downloads run one file at a time, resume after interruption and are verified with SHA-256. Once
installed, generation runs offline. [MODELS.md](MODELS.md) lists every package and how to
download it by hand.

### Windows / NVIDIA

Use 64-bit Windows, an NVIDIA GPU with a current CUDA 13-capable driver, Python 3.11 (with the
`py` launcher) and FFmpeg/FFprobe on PATH. Run
`powershell -ExecutionPolicy Bypass -File "Install OpenSuno.ps1"`, then
`powershell -ExecutionPolicy Bypass -File "Launch Studio.ps1"` and open <http://127.0.0.1:7862>.
In Models download **PyTorch CUDA BF16**, the Windows default: the official YuE2-3B checkpoint run by
the upstream `yue2-infer` package in an isolated `.cuda-venv` (PyTorch 2.10 / CUDA 12.8).
**PyTorch CUDA FP8** shares those downloads and quantizes token generation only; it was slower in
testing. MLX BF16 stays available for comparison.

The CUDA adapter (`cuda_engine.py`, not a fork of upstream) reserves 2 GiB of VRAM, offloads
token-generation layers during synthesis and decodes audio in 512-frame tiles; on Windows it selects
cuDNN attention so SDPA never falls back to a large math-attention buffer. Seeds are repeatable
within a backend, not across Metal and CUDA. Verify with
`.venv\Scripts\python.exe check_runtime.py` and
`.venv\Scripts\python.exe -m unittest discover -s tests -v`. Treat Windows as experimental until
your song lengths and cover workflow have been validated.

## Render on another machine

The Studio (interface, library, OpenAI helpers) and the GPU can be separate machines. The GPU
machine runs the **render node**, a small API that receives a job, downloads the model it needs,
renders, reports progress and keeps the finished song for streaming.

On a Mac the disk image installs the **OpenSuno Render Node** menu bar app (start/stop through
launchd, current song and speed, the address and token other Studios need, Start at login). From a
terminal:

```
OPENSUNO_NODE_TOKEN=choose-a-secret "./Launch Render Node.command"   # macOS
$env:OPENSUNO_NODE_TOKEN = 'choose-a-secret'; .\"Launch Render Node.ps1"   # Windows
```

The node listens on port 7863 (`OPENSUNO_NODE_PORT`); without a token it only accepts its own
machine. It keeps loaded models in memory and releases them after 30 idle minutes
(`OPENSUNO_NODE_IDLE_MINUTES`). In the Studio, **Settings → Rendering**: address, token,
**Connect**. Models are downloaded on the node; a song keeps rendering if the Studio disconnects and
the Studio reattaches when it is back. `OPENSUNO_RENDER_URL` / `OPENSUNO_RENDER_TOKEN` fix the node
and hide the setting.

The audio stays on the node: it encodes an MP3 next to the WAV, the Studio fetches only the small
files (tokens, plan, result) and streams the MP3. **Download** brings the WAV and MP3 into the
library for keeps. The node keeps finished jobs for 30 days (`OPENSUNO_NODE_KEEP_DAYS`).

Node API (`/v1`, bearer token): `GET health`, `POST models/download?model=`, `POST jobs`
(multipart: `request` JSON, `inputs` side files, `uploads` recordings), `GET jobs/{id}?log_offset=`,
`GET jobs/{id}/files/{name}`, `POST jobs/{id}/cancel`, `DELETE jobs/{id}`.

## What you can do

Details for each are in [FEATURES.md](FEATURES.md).

- **Create** a song from lyrics and a style. **Add Style** offers 73 presets (from SongScribe,
  MIT) with blending and modifiers; the sparkles button asks OpenAI for a style line (key in
  **Settings**, `OPENAI_API_KEY`, or a local `.openai-key` file; never committed). Songs get album
  art from OpenAI when a key is set.
- **Cover / Remix** an uploaded recording: trim, analyse, keep the melody or melody and chords, and
  let the model **arrange** the rest in the new style. **Vocal Range** moves the melody to the
  chosen voice. **Continue this recording** keeps the audio up to a point and writes what follows.
- **Finished songs**: re-render with more audio steps, **Extend** or **Regenerate from here** at a
  bar, **Edit** the arrangement (style, tempo, harmony, structure) at the score level, **Adjust
  Speed** (ffmpeg, with or without keeping pitch), favorites, rename, WAV/MP3 downloads, projects.
- **LoRA adapters**: community YuE2 LoRAs in `loras/` (HF, PEFT or ComfyUI layout) appear under
  **More Options → LoRA** with a strength; applied to that song only and saved with it. See
  `loras/README.md`.
- **Find lyrics** searches LRCLIB by title and artist; tags embedded in an uploaded file prefill the
  title and lyrics.
- Songs are queued while one renders; models stay resident between songs and are released after the
  last page closes.

Logs are in `~/Library/Logs/OpenSuno/server.log` on a Mac. The server listens only on `127.0.0.1`;
recordings and results stay in `uploads/` and `library/`.

## Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `YUE2_PORT` | Studio | Port of the browser page (default `7862`). |
| `OPENSUNO_DATA` | Studio | Folder for `library/`, `uploads/` and caches (default: the checkout). |
| `OPENSUNO_HOSTED` | Studio | Hosted mode: no local models, jobs go to a render node. |
| `OPENSUNO_RENDER_URL`, `OPENSUNO_RENDER_TOKEN` | Studio | Fix the render node; the Settings form is then read-only. |
| `OPENSUNO_NODE_PORT`, `OPENSUNO_NODE_HOST`, `OPENSUNO_NODE_TOKEN` | Render node | Port (default `7863`), bind address and shared token. |
| `OPENSUNO_NODE_IDLE_MINUTES`, `OPENSUNO_NODE_KEEP_DAYS` | Render node | Unload models when idle; delete finished audio after this many days. |
| `OPENSUNO_HOME`, `OPENSUNO_APPLICATIONS`, `OPENSUNO_INSTALL_START` | Mac installer | Install folder, Applications folder, and `0` to skip launching after install. |
| `OPENAI_API_KEY` | Studio / node | Key for Ask AI and album art when none is saved in Settings. |

## What is in Git

Application code, model runtime source, small configuration files, pinned dependency lists and the
download manifest. No model weights, tokenizers, recordings, generated songs, environments or caches
are tracked.

## Sources

- [YuE2](https://huggingface.co/m-a-p/YuE2-3B)
- [Community MLX port](https://huggingface.co/ahmadw/YuE2-3B-MLX), revision
  `fe0a9050fd658257b486b880422d8872ee1f81e3`
- [SheetSage2](https://huggingface.co/m-a-p/SheetSage2), revision
  `eab522a8168e8b8b8c4856bf8609cd86198f01fe`
- [MERT-v2](https://huggingface.co/m-a-p/MERT-v2-FullSong), revision
  `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`
- [yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)
  (CC BY-NC 4.0) for Audio input (v9 head and matching NAR adapter) and the Real-audio decoder v9 LoRA card
- [SongScribe](https://github.com/TheLocalLab/ComfyUI-SongScribe) (MIT, Copyright (c) 2026
  TheLocalLab): the 73 style presets and the lyrics-tag approach, adapted in `web/style-presets*.js`
  and `web/lyrics-structure.js`

Upstream attribution and license notices are retained with the runtime source. YuE2 model weights
are CC BY-NC 4.0 (non-commercial); upstream terms apply when downloaded separately.

## License

OpenSuno's own code is [MIT](LICENSE). Code adapted from SheetSage2, MERT-v2 and the YuE2 MLX port
(`transcriber/`, `mert/`, parts of `model/`) stays CC BY-NC 4.0, and the model weights are
CC BY-NC 4.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Development checks

`.venv/bin/python -m unittest discover -s tests -v` and `for f in tests/*.cjs; do node "$f"; done`.
The tests use tiny local HTTP fixtures and fake models; nothing is downloaded.
