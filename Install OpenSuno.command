#!/bin/bash
set -euo pipefail
OPEN_SUNO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$OPEN_SUNO_DIR"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "OpenSuno currently requires an Apple Silicon Mac. The MLX backend does not support Windows/NVIDIA or Intel Macs."
  exit 1
fi
if ! command -v brew >/dev/null; then
  echo "Install Homebrew from https://brew.sh, then run this installer again."
  exit 1
fi
echo "Installing Python and FFmpeg prerequisites…"
brew install python@3.12 python@3.11 ffmpeg
PY_GEN="$(brew --prefix python@3.12)/bin/python3.12"
PY_COVER="$(brew --prefix python@3.11)/bin/python3.11"
[[ -x .venv/bin/python ]] || "$PY_GEN" -m venv .venv
[[ -x .transcribe-venv/bin/python ]] || "$PY_COVER" -m venv .transcribe-venv
.venv/bin/python -m pip install -r requirements-installed.txt
.transcribe-venv/bin/python -m pip install -r requirements-transcriber.txt
.venv/bin/python -c 'import fastapi, uvicorn, mlx.core, requests, soundfile, tiktoken, multipart'
.transcribe-venv/bin/python -c 'import torch, torchaudio, transformers, scipy, pretty_midi'
echo "OpenSuno dependencies are ready. Open Launch Studio.command, then open Models and download BF16. Cover analysis and Audio input are separate options."
