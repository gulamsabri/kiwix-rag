#!/bin/bash
# batch_index.sh — extract and index a queue of ZIM files from a manifest.
#
# Resumable: skips extraction if .jsonl already exists, skips indexing if
# a .indexed marker file exists next to the .jsonl.
#
# Manifest format (see batch_manifest.example):
#   - One ZIM stem per line (filename without .zim)
#   - Append --ocr to enable OCR for scanned PDFs
#   - Lines starting with # and blank lines are ignored
#
# Usage:
#   bash batch_index.sh batch_manifest.conf
#   bash batch_index.sh batch_manifest.conf 2>&1 | tee batch_live.log

set -euo pipefail

MANIFEST="${1:-}"
if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
    echo "Usage: $0 <manifest_file>" >&2
    echo "See batch_manifest.example for the expected format." >&2
    exit 1
fi

KIWIX_DIR="${KIWIX_DIR:-/Volumes/Extreme SSD/kiwix-library}"
VENV="${VENV:-$HOME/kiwix-rag/bin/activate}"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPTS/batch_$(date +%Y%m%d_%H%M%S).log"

source "$VENV"
export HF_HUB_OFFLINE=1

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

process_zim() {
    local name="$1"
    local ocr_flag="${2:-}"
    local zim="$KIWIX_DIR/${name}.zim"
    local jsonl="$KIWIX_DIR/${name}_chunks.jsonl"
    local marker="${jsonl}.indexed"

    if [ ! -f "$zim" ]; then
        log "SKIP $name — ZIM not found"
        return
    fi

    log "━━━ $name ━━━"

    if [ -f "$jsonl" ]; then
        log "  extraction: already done ($(wc -l < "$jsonl" | tr -d ' ') chunks)"
    else
        log "  extraction: starting${ocr_flag:+ $ocr_flag}"
        python "$SCRIPTS/extract_zim.py" "$zim" $ocr_flag >> "$LOG" 2>&1
        if [ -f "$jsonl" ]; then
            log "  extraction: done ($(wc -l < "$jsonl" | tr -d ' ') chunks)"
        else
            log "  extraction: FAILED — skipping indexing"
            return
        fi
    fi

    if [ -f "$marker" ]; then
        log "  indexing: already done"
    else
        log "  indexing: starting"
        python "$SCRIPTS/build_index.py" "$jsonl" >> "$LOG" 2>&1
        touch "$marker"
        log "  indexing: done"
    fi
}

log "════ Batch started: $MANIFEST ════"
log "ZIM dir:  $KIWIX_DIR"
log "Log file: $LOG"
log ""

while IFS= read -r line || [ -n "$line" ]; do
    # strip leading/trailing whitespace
    line="${line#"${line%%[! ]*}"}"
    line="${line%"${line##*[! ]}"}"

    # skip blank lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue

    # split into name and optional flags (e.g. "--ocr")
    read -r name flags <<< "$line"
    process_zim "$name" "${flags:-}"
done < "$MANIFEST"

log ""
log "════ Batch complete ════"
