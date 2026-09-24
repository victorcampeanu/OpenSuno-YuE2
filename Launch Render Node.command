#!/bin/bash
# Starts this Mac as an OpenSuno render node: a Studio on another machine sends songs here to render.
# Set OPENSUNO_NODE_TOKEN to accept connections from other machines (without it, only this Mac can connect).
set -e
YUE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$YUE_DIR"
export PATH="$YUE_DIR/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "The Mac render node requires an Apple Silicon Mac (MLX)."
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  /bin/bash "$YUE_DIR/Install OpenSuno.command"
fi
if [[ -z "$OPENSUNO_NODE_TOKEN" ]]; then
  echo "OPENSUNO_NODE_TOKEN is not set: only this Mac can connect. To serve other machines, run:"
  echo "  OPENSUNO_NODE_TOKEN=choose-a-secret \"$YUE_DIR/Launch Render Node.command\""
fi
echo "Keep this window open; Ctrl+C stops the render node."
exec "$YUE_DIR/.venv/bin/python" "$YUE_DIR/render_node.py"
