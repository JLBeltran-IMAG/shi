#!/bin/bash
set -e

SCRIPT_NAME="shi.py"
TOOL_NAME="shi"

# Absolute path of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install python package
python3 -m pip install "$SCRIPT_DIR"

# Optional: editable mode (dev)
# python3 -m pip install -e "$SCRIPT_DIR"

echo "Python package installed successfully."

# If you still want a symlink (not needed if you use entry points)
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

if [ -f "$SCRIPT_PATH" ]; then
    chmod +x "$SCRIPT_PATH"
    sudo ln -sf "$SCRIPT_PATH" /usr/local/bin/$TOOL_NAME
fi

echo "Software '$TOOL_NAME' is now available in your command-line interface"
