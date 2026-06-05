#!/bin/bash
# Wrapper for kiwix-serve so systemd can use a glob for ZIM files.
exec /usr/bin/kiwix-serve --port 8080 /mnt/ssd/kiwix-library/*.zim
