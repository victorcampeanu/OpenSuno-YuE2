#!/bin/bash
set -e
YUE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$YUE_DIR"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "OpenSuno currently requires an Apple Silicon Mac (MLX)."
  exit 1
fi
if curl -fsS http://127.0.0.1:7862/api/config >/dev/null 2>&1; then
  open http://127.0.0.1:7862
  exit 0
fi
if [[ ! -x .venv/bin/python || ! -x .transcribe-venv/bin/python ]]; then
  /bin/bash "$YUE_DIR/scripts/Install OpenSuno.command"
fi
exec "$YUE_DIR/.venv/bin/python" "$YUE_DIR/launcher/launch.py"
