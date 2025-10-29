#!/usr/bin/env bash
#MISE description="Clean build outputs"

set -eo pipefail

if [[ -n "$(git status -u --porcelain)" ]]; then
    echo "Git status is not clean, cannot clean." >&2
    exit 3
fi

set -x
rm -rf build dist output target
rm -f *.lprof *.profraw *.prof *.profdata lcov.info coverage.* .coverage *.log
git clean -xf docs src
