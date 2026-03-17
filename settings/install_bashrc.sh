#!/usr/bin/env bash
# Append terminal beautification source line to ~/.bashrc if not already present.
# Usage: bash settings/install_bashrc.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASHRC_SH="$SCRIPT_DIR/bashrc.sh"
SOURCE_LINE="[ -f \"$BASHRC_SH\" ] && source \"$BASHRC_SH\""

if grep -qF "bashrc.sh" ~/.bashrc 2>/dev/null; then
    echo "Already installed in ~/.bashrc, skipping."
else
    cat >> ~/.bashrc << EOF

# ── Terminal beautification (prompt + history) ───────────────────────────────
$SOURCE_LINE
# ── End terminal beautification ──────────────────────────────────────────────
EOF
    echo "Installed. Run 'source ~/.bashrc' to activate."
fi
