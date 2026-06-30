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
#   bash update_pi.sh                 # (no-op without flags; Postgres data lives on the Pi)
#   bash update_pi.sh --scripts       # also sync web.py / templates / eval.py
#   bash update_pi.sh --kiwix         # also rebuild kiwix library + restart kiwix-serve
#   bash update_pi.sh --scripts --kiwix
#   bash update_pi.sh --services      # also deploy systemd service files
#
# Typical workflow for adding a new ZIM:
#   1. scp pi@meshpi.local:/mnt/ssd/kiwix-library/new.zim ~/Downloads/
#   2. source ~/kiwix-rag/bin/activate
#   3. python ~/kiwix-rag-project/extract_zim.py ~/Downloads/new.zim -o ~/Downloads/new_chunks.jsonl
#   4. python ~/kiwix-rag-project/build_index.py ~/Downloads/new_chunks.jsonl
#   5. Add the collection name pattern to the right GROUPS entry in web.py
#   6. Connect SSD to Mac, run: bash ~/kiwix-rag-project/update_pi.sh --scripts

set -euo pipefail

SSD="${SSD:-/Volumes/Extreme SSD}"
PI="${PI:-pi@meshpi.local}"
SCRIPTS_SRC="$HOME/kiwix-rag-project"
SCRIPTS_DEST="$SSD/kiwix-rag-project"

sync_scripts=false
rebuild_kiwix=false
sync_services=false

for arg in "$@"; do
    case "$arg" in
        --scripts)    sync_scripts=true ;;
        --kiwix)      rebuild_kiwix=true ;;
        --services)   sync_services=true ;;
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

# ── sync scripts ──────────────────────────────────────────────────────────────

if $sync_scripts; then
    echo "━━━ Syncing scripts ━━━"
    for f in rag.py web.py eval.py pg_client.py migrate_chroma_to_pg.py verify_migration.py; do
        cp "$SCRIPTS_SRC/$f" "$SCRIPTS_DEST/$f" && echo "  $f"
    done
    rsync -ah --delete \
        "$SCRIPTS_SRC/templates/" \
        "$SCRIPTS_DEST/templates/"
    echo "  templates/"
    cp "$SCRIPTS_SRC/kiwix-rag.service" "$SCRIPTS_DEST/kiwix-rag.service" && echo "  kiwix-rag.service"
    echo ""
fi

# ── sync systemd services ─────────────────────────────────────────────────────

if $sync_services; then
    echo "━━━ Syncing systemd services ━━━"
    for f in kiwix-rag.service kiwix-serve.service caddy-kiwix.service; do
        if [ -f "$SCRIPTS_SRC/$f" ]; then
            cp "$SCRIPTS_SRC/$f" "$SCRIPTS_DEST/$f" && echo "  $f → $SCRIPTS_DEST/"
        fi
    done
    # Also stage Caddy config if present
    if [ -f "$SCRIPTS_SRC/Caddyfile" ]; then
        cp "$SCRIPTS_SRC/Caddyfile" "$SCRIPTS_DEST/Caddyfile" && echo "  Caddyfile → $SCRIPTS_DEST/"
    fi
    echo "  Services staged on SSD. Deploy on the Pi after reconnecting:"
    echo "    ssh $PI 'sudo cp ~/kiwix-rag-project/kiwix-rag.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl restart kiwix-rag'"
    echo ""
fi

# ── done — remind user to reconnect SSD ──────────────────────────────────────

echo "━━━ Sync complete ━━━"
echo "Eject the SSD from this Mac, reconnect to the Pi, then:"
if $rebuild_kiwix; then
    echo "  ssh $PI 'bash ~/build_kiwix_library.sh && sudo systemctl restart kiwix-rag kiwix-serve'"
    if $sync_scripts; then
        echo ""
        echo "  If the service file changed, also run first:"
        echo "  ssh $PI 'sudo cp /mnt/ssd/kiwix-rag-project/kiwix-rag.service /etc/systemd/system/ && sudo systemctl daemon-reload'"
    fi
else
    echo "  ssh $PI 'sudo systemctl restart kiwix-rag'"
    if $sync_scripts; then
        echo ""
        echo "  If the service file changed, also run:"
        echo "  ssh $PI 'sudo cp /mnt/ssd/kiwix-rag-project/kiwix-rag.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl restart kiwix-rag'"
    fi
fi
