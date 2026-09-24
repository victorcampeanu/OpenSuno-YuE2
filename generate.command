#!/bin/bash
set -e
YUE_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$YUE_DIR/.venv/bin/python" "$YUE_DIR/model/generate.py" --model "$YUE_DIR/model/bf16" "$@"
