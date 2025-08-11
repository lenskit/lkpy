VERBOSE="${VERBOSE:-0}"

dbg() {
    if (($VERBOSE)); then
        echo "$@" >&2
    fi
}

msg() {
    echo "$@" >&2
}

echo-run() {
    echo "+ $*" >&2
    "$@"
    return "$?"
}

git-is-clean() {
    if [[ -z "$(git status -u --porcelain)" ]]; then
        return 0
    else
        return 1
    fi
}
