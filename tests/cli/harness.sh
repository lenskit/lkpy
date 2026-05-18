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

run-command() {
    local status
    local ctext="$(echo "$*" | sed -e "s@$TEST_WORK@\$TEST_WORK@g" -e "s@$TEST_DIR@\$TEST_DIR@g")"
    echo "+ $ctext"
    "$@"
    status="$?"
    if (($status)); then
        not_ok "exit($status): $ctext"
        echo "# command failed: $ctext" >&5
    else
        ok "$ctext"
    fi
    return $status
}

run-python() {
    run-command $PYRUN "$@"
    return "$?"
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
        tap_out "$status $N $*"
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
        not_ok "require $*"
    fi
}

if grep -q '^begin-suite' "$TEST"; then
    dbg "test will start itself"
else
    dbg "auto-starting suite"
    begin-suite
fi
. "$TEST"
