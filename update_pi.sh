#!/bin/bash
# update_pi.sh — sync updated content to the Pi via the Extreme SSD.
#
# Workflow:
#   1. Stop Pi services (kiwix-rag needs the SSD)
#   2. Safely eject SSD from Pi, connect to Mac
#   3. Run this script
#   4. Eject SSD from Mac, reconnect to Pi
#   5. Run the SSH command printed at the end of this script to push the
#      updated vector_db to its live location and restart services
#
# Usage:
#   bash update_pi.sh                 # sync vector DB only
#   bash update_pi.sh --scripts       # also sync web.py / templates / eval.py
#   bash update_pi.sh --kiwix         # also rebuild kiwix library + restart kiwix-serve
#   bash update_pi.sh --scripts --kiwix
#
# Typical workflow for adding a new ZIM:
#   1. scp pi@meshpi.local:/mnt/ssd/kiwix-library/new.zim ~/Downloads/
#   2. source ~/kiwix-rag/bin/activate
#   3. python ~/kiwix-rag-project/extract_zim.py ~/Downloads/new.zim -o ~/Downloads/new_chunks.jsonl
#   4. python ~/kiwix-rag-project/build_index.py ~/Downloads/new_chunks.jsonl
#   5. Add the collection name pattern to the right GROUPS entry in web.py
#   6. Connect SSD to Mac, run: bash ~/kiwix-rag-project/update_pi.sh --scripts

set -euo pipefail

SSD="/Volumes/Extreme SSD"
PI="pi@meshpi.local"
SCRIPTS_SRC="$HOME/kiwix-rag-project"
SCRIPTS_DEST="$SSD/kiwix-rag-project"

# Destination path for vector_db on the Pi (overridable via env var)
PI_DB_DEST="${PI_DB_DEST:-/mnt/nvme/vector_db}"

sync_scripts=false
rebuild_kiwix=false

for arg in "$@"; do
    case "$arg" in
        --scripts) sync_scripts=true ;;
        --kiwix)   rebuild_kiwix=true ;;
    esac
done

# ── preflight ─────────────────────────────────────────────────────────────────

if [ ! -d "$SSD" ]; then
    echo "Error: SSD not mounted at '$SSD'" >&2
    echo "Connect the Extreme SSD to this Mac first." >&2
    exit 1
fi

echo "Target SSD: $SSD"
echo ""

# ── sync vector DB ────────────────────────────────────────────────────────────

echo "━━━ Syncing vector DB ━━━"
rsync -ah --progress \
    "$SCRIPTS_SRC/vector_db/" \
    "$SSD/vector_db/"
echo ""

# ── sync scripts ──────────────────────────────────────────────────────────────

if $sync_scripts; then
    echo "━━━ Syncing scripts ━━━"
    for f in rag.py web.py eval.py requirements.txt; do
        cp "$SCRIPTS_SRC/$f" "$SCRIPTS_DEST/$f" && echo "  $f"
    done
    rsync -ah --delete \
        "$SCRIPTS_SRC/templates/" \
        "$SCRIPTS_DEST/templates/"
    echo "  templates/"
    echo ""
fi

# ── done — remind user to reconnect SSD ──────────────────────────────────────

echo "━━━ Sync complete ━━━"
echo "Eject the SSD from this Mac, reconnect to the Pi, then:"
if $rebuild_kiwix; then
    echo "  ssh $PI 'rsync -a --delete /mnt/ssd/vector_db/ $PI_DB_DEST/ && bash ~/build_kiwix_library.sh && sudo systemctl restart kiwix-rag kiwix-serve'"
else
    echo "  ssh $PI 'rsync -a --delete /mnt/ssd/vector_db/ $PI_DB_DEST/ && sudo systemctl restart kiwix-rag'"
fi
