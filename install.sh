#!/bin/bash

SCRIPT_NAME="shi.py"
TOOL_NAME="shi"

# Current path
SCRIPT_PATH="$PWD/$SCRIPT_NAME"

# Verify file exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: El archivo $SCRIPT_PATH no existe en el directorio actual."
    exit 1
fi

# Made it executable
chmod +x "$SCRIPT_PATH"

# Simbolic link to /usr/local/bin (superuser)
TARGET_DIR="/usr/local/bin"
sudo ln -sf "$SCRIPT_PATH" "$TARGET_DIR/$TOOL_NAME"

echo "Software '$TOOL_NAME' is now available in your command-line interface"
