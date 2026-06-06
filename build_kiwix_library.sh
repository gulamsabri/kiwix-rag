#!/bin/bash
# build_kiwix_library.sh — scan ZIM files and build a kiwix library XML,
# skipping any files that kiwix-manage can't load.

LIBRARY="/mnt/ssd/library.xml"
LIBRARY_TMP="${LIBRARY}.tmp"
ZIM_DIR="/mnt/ssd/kiwix-library"

echo "Building Kiwix library XML..."
rm -f "$LIBRARY_TMP"

ok=0
skip=0
for f in "$ZIM_DIR"/*.zim; do
    [ -f "$f" ] || continue
    if kiwix-manage "$LIBRARY_TMP" add "$f" 2>/dev/null; then
        ok=$((ok + 1))
        echo -n "."
    else
        skip=$((skip + 1))
        echo ""
        echo "  SKIP: $(basename "$f")"
    fi
done

echo ""
echo "Done: $ok ZIM(s) added, $skip skipped."
# Atomically replace the live library only after a successful build
mv "$LIBRARY_TMP" "$LIBRARY"
echo "Library written to $LIBRARY"
