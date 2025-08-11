#!/bin/bash
#MISE description="Initialize output directories."

OUT_DIRS=(build output dist)

set -euo pipefail

for dir in "$OUT_DIRS[@]"; do
    mkdir -p "$dir"
    if [[ ! -f "$dir/CACHEDIR.TAG" ]]; then
        cat >"$dir/CACHEDIR.TAG" <<TAG
Signature: 8a477f597d28d172789f06886806bc55
# LensKit output directory, exclude from backups.
TAG
    fi
done
