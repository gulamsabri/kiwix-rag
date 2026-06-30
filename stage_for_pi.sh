#!/bin/bash
# stage_for_pi.sh — copy Pi 5 runtime files to the Extreme SSD
#
# Run this after the overnight batch completes.
# Extraction, indexing, and OCR stay on the Mac; the Pi only needs
# the vector DB, the query script, and the two models.
#
# Usage:
#   bash ~/kiwix-rag-project/stage_for_pi.sh

set -euo pipefail

SSD="/Volumes/Extreme SSD"
DEST_MODELS="$SSD/ollama-models"
DEST_HF="$SSD/hf-cache"
DEST_SCRIPTS="$SSD/kiwix-rag-project"

SCRIPTS_SRC="$HOME/kiwix-rag-project"
SRC_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"   # macOS default; Linux is ~/.local/ollama/models
SRC_HF="$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
SRC_SCRIPT="$SCRIPTS_SRC/rag.py"
SRC_WEB="$SCRIPTS_SRC/web.py"
SRC_TEMPLATES="$SCRIPTS_SRC/templates"

# ── preflight ─────────────────────────────────────────────────────────────────

if [ ! -d "$SSD" ]; then
    echo "Error: SSD not mounted at '$SSD'" >&2
    exit 1
fi

df -h "$SSD"
echo ""

# ── helpers ───────────────────────────────────────────────────────────────────

sync_dir() {
    local label="$1" src="$2" dest="$3"
    local src_size
    src_size=$(du -sh "$src" 2>/dev/null | cut -f1)
    echo "━━━ $label ($src_size) ━━━"
    echo "  $src"
    echo "  → $dest"
    mkdir -p "$dest"
    rsync -ah --progress "$src/" "$dest/"
    echo ""
}

# ── copy ──────────────────────────────────────────────────────────────────────

echo "Staging Pi 5 runtime files to Extreme SSD..."
echo ""

sync_dir "Ollama models"   "$SRC_MODELS"  "$DEST_MODELS"
sync_dir "Embedding model" "$SRC_HF"      "$DEST_HF/models--sentence-transformers--all-MiniLM-L6-v2"

echo "━━━ scripts ━━━"
mkdir -p "$DEST_SCRIPTS"
cp "$SRC_SCRIPT" "$DEST_SCRIPTS/rag.py"
cp "$SRC_WEB"    "$DEST_SCRIPTS/web.py"
cp "$HOME/kiwix-rag-project/pg_client.py"        "$DEST_SCRIPTS/pg_client.py"
cp "$HOME/kiwix-rag-project/migrate_chroma_to_pg.py" "$DEST_SCRIPTS/migrate_chroma_to_pg.py"
cp "$HOME/kiwix-rag-project/verify_migration.py"  "$DEST_SCRIPTS/verify_migration.py"
rsync -ah "$SRC_TEMPLATES/" "$DEST_SCRIPTS/templates/"
echo "  Copied rag.py, web.py, pg_client.py, migrate_chroma_to_pg.py, verify_migration.py, templates/ to $DEST_SCRIPTS/"
echo ""

# ── summary ───────────────────────────────────────────────────────────────────

echo "Done. Files on SSD:"
echo "  $DEST_MODELS"
echo "  $DEST_HF"
echo "  $DEST_SCRIPTS/{rag.py,web.py,pg_client.py,migrate_chroma_to_pg.py,verify_migration.py,templates/}"
echo ""
echo "On the Pi — install deps (first time only), start Ollama, then the web UI:"
echo "  ~/kiwix-rag/bin/pip install -r /mnt/ssd/kiwix-rag-project/requirements.txt"
echo "  OLLAMA_MODELS=/mnt/ssd/ollama-models ollama serve &"
echo "  HF_HUB_OFFLINE=1 HF_HUB_CACHE=/mnt/ssd/hf-cache \\"
echo "    python /mnt/ssd/kiwix-rag-project/web.py --dsn postgresql://kiwix@/kiwix_rag"
echo ""
echo "Then open http://<pi-hostname>:5000 in a browser."
