#!/bin/bash
# Builds dist/OpenSuno.dmg: one installer that puts the Studio, the render node or both on a Mac with nothing installed.
set -e
YUE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$YUE_DIR"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools are needed to compile the apps: run  xcode-select --install  and try again."
  exit 1
fi
exec /usr/bin/python3 "$YUE_DIR/launcher/build_installer.py" "$@"
