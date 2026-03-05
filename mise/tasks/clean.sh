#!/usr/bin/env bash
#MISE description="Clean build outputs"
#USAGE flag "-d --docs" help="Clean documentation output and generated files."

typeset -A clean_parts=()

_want_clean() {
    local what="$1"
    if ((${clean_parts[$what]:-0})); then
        # we want to clean this
        return 0
    elif ((${#clean_parts[@]} == 0)); then
        # we want to clean everything
        return 0
    else
        # we don't want to clean this
        return 1
    fi
}

if [[ $usage_docs ]]; then
    clean_parts[docs]=1
fi

if _want_clean build; then
    echo "cleaning build and distribution outputs" 2>&1
    rm -rf build dist output target
fi

if _want_clean docs; then
    if [[ -n "$(git status -u --porcelain)" ]]; then
        echo "Git status is not clean, cannot clean docs." >&2
        exit 3
    fi
    echo "cleaning build/site" 2>&1
    rm -rf build/site
    echo "cleaning generated doc files"
    git clean -xf docs src
fi

if _want_clean logs; then
    echo "cleaning profiles, logs, etc." 2>&1
    rm -rf *.lprof *.profraw *.prof *.profdata .coverage-prof lcov.info coverage.* .coverage *.log
fi
