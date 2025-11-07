#!/usr/bin/env bash
# Run a single test.

. "$TEST_DIR/../../mise/task-functions.sh"
TEST="$1"
N=0

if [[ -z $TEST_WORK ]]; then
    die "missing TEST_WORK env var"
fi
if [[ -z $TEST_DIR ]]; then
    die "missing TEST_DIR env var"
fi
if [[ -z $ML_TEST_DIR ]]; then
    die "missing ML_TEST_DIR env var"
fi

begin-suite() {
    perl "$TEST_DIR/count-tests.pl" "$TEST" >&5
}

run-python() {
    local status
    echo "+ python $*"
    $PYRUN "$@"
    status="$?"
    if (($status)); then
        not_ok "$@"
        echo "# command failed: $PYRUN $0" >&5
    else
        ok "$@"
    fi
    return $status
}

run-lenskit() {
    run-python -m lenskit "$@"
    return "$?"
}

tap_out() {
    echo "$@" >&5
}

tap_status() {
    local status="$1"
    shift
    N=$(($N + 1))
    if [[ $1 ]]; then
        tap_out "$status $N - $*"
    else
        tap_out "$status $N"
    fi
}

tap_comment() {
    tap_out "# $*"
}

ok() {
    tap_status ok "$@"
}

not_ok() {
    tap_status "not ok" "$@"
}

require() {
    if test "$@"; then
        ok
    else
        not_ok
        tap_comment "require $*"
    fi
}

if ! grep -qv '^begin-suite' "$TEST"; then
    begin-suite
fi
. "$TEST"
