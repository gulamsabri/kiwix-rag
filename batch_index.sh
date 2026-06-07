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

process_zim_single() {
    local name="$1"
    local extract_flags="${2:-}"
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
        log "  extraction: starting${extract_flags:+ $extract_flags}"
        python "$SCRIPTS/extract_zim.py" "$zim" $extract_flags >> "$LOG" 2>&1
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
        if python "$SCRIPTS/build_index.py" "$jsonl" >> "$LOG" 2>&1; then
            touch "$marker"
            log "  indexing: done"
        else
            log "  indexing: FAILED — collection may already exist; see log for details"
        fi
    fi
}

process_zim_parts() {
    local name="$1"
    local part_size="$2"
    local max_parts="${3:-0}"   # 0 = unlimited
    local extract_flags="${4:-}"
    local zim="$KIWIX_DIR/${name}.zim"

    if [ ! -f "$zim" ]; then
        log "SKIP $name — ZIM not found"
        return
    fi

    local total
    total=$(python -c "import libzim; print(libzim.Archive('$zim').all_entry_count)")
    local limit_note=""
    [ "$max_parts" -gt 0 ] && limit_note=" (max $max_parts new parts this run)"
    log "━━━ $name (part mode: ${part_size} entries/part, ${total} total${limit_note}) ━━━"

    local offset=0
    local part=1
    local new_parts=0
    while [ "$offset" -lt "$total" ]; do
        local suffix
        suffix=$(printf "e%08d" "$offset")
        local jsonl="$KIWIX_DIR/${name}_${suffix}_chunks.jsonl"
        local marker="${jsonl}.indexed"

        if [ -f "$marker" ]; then
            log "  part $part (offset $offset): already indexed"
        else
            # Honour --max-parts: stop once we've done enough new work this run
            if [ "$max_parts" -gt 0 ] && [ "$new_parts" -ge "$max_parts" ]; then
                log "  part $part (offset $offset): max-parts ($max_parts) reached — stopping for this run"
                return
            fi
            log "  part $part (offset $offset): extracting${extract_flags:+ $extract_flags}..."
            python "$SCRIPTS/extract_zim.py" "$zim" \
                --entry-offset "$offset" --entry-limit "$part_size" \
                $extract_flags >> "$LOG" 2>&1
            if [ ! -f "$jsonl" ]; then
                log "  part $part: extraction FAILED — stopping"
                return
            fi
            local chunk_count
            chunk_count=$(wc -l < "$jsonl" | tr -d ' ')
            log "  part $part: extraction done ($chunk_count chunks)"
            if [ "$chunk_count" -eq 0 ]; then
                log "  part $part: no chunks — skipping indexing"
                touch "$marker"
            else
                log "  part $part: indexing..."
                if python "$SCRIPTS/build_index.py" "$jsonl" >> "$LOG" 2>&1; then
                    touch "$marker"
                    log "  part $part: indexed"
                else
                    log "  part $part: indexing FAILED — see log for details"
                fi
            fi
            new_parts=$(( new_parts + 1 ))
        fi

        offset=$(( offset + part_size ))
        part=$(( part + 1 ))
    done

    log "  all $((part - 1)) parts complete"
}

process_zim() {
    local name="$1"
    local raw_flags="${2:-}"

    # Parse --part-size N and --max-parts N out of flags; pass remainder to extract_zim.py
    local part_size=0
    local max_parts=0
    local extract_flags=()
    local next_is=""
    local token
    for token in $raw_flags; do
        case "$next_is" in
            part-size) part_size="$token"; next_is="" ;;
            max-parts) max_parts="$token"; next_is="" ;;
            *)
                case "$token" in
                    --part-size) next_is="part-size" ;;
                    --max-parts) next_is="max-parts" ;;
                    *)           extract_flags+=("$token") ;;
                esac
                ;;
        esac
    done

    if [ "$part_size" -gt 0 ] 2>/dev/null; then
        process_zim_parts "$name" "$part_size" "$max_parts" "${extract_flags[*]:-}"
    else
        process_zim_single "$name" "$raw_flags"
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
