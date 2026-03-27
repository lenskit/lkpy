if [[ -z $VERBOSE ]]; then
    if [[ $usage_verbose == true ]]; then
        VERBOSE=1
    else
        VERBOSE=0
    fi
fi

dbg() {
    if (($VERBOSE)); then
        echo "$@" >&2
    fi
}

msg() {
    echo "$@" >&2
}

err() {
    echo "ERROR: $*" >&2
}

die() {
    local ec=2
    if [[ "$1" =~ '^-[0-9]$' ]]; then
        ec=${1#-}
        shift
    fi
    err "$@"
    exit $ec
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
