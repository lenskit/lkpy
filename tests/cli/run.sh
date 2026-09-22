#!/usr/bin/env bash

. "$PWD/scripts/lib/init.sh" || exit 2
export TEST_DIR="$(dirname "$0")"

_cli_definition() {
    setup POSARGS help:_cli_help -- "Usage: run.sh [options] [TEST...]" ''
    msg 'Options:'
    flag LOG_VERBOSE init:@export -v --verbose -- 'output verbose log messages'
    flag COV_RECORD init:=0 --coverage -- 'record test coverage'
    flag COV_APPEND init:=0 --cov-append -- 'append to existing test coverage'
    disp :_cli_help -h --help -- 'display script help'
}
eval "$(getoptions _cli_definition _cli_parse) exit 1"
_cli_parse "$@"
eval "set -- $POSARGS"

start-group() {
    if [[ $CI_SYSTEM_NAME = woodpecker ]]; then
        step "$*"
    elif [[ $CI ]]; then
        echo "::group::$*"
    fi
}

end-group() {
    if [[ $CI ]]; then
        echo ::endgroup::
    fi
}

PYRUN="python"
if (($COV_RECORD)); then
    PYRUN="coverage run -a"

    if (($COV_APPEND)); then
        msg "appending to test coverage"
    else
        msg "resetting test coverage"
        coverage erase || die "failed to erase coverage"
    fi
fi
export PYRUN

export ML_TEST_DIR="data/ml-latest-small"
if [[ ! -f "$ML_TEST_DIR/ratings.csv" ]]; then
    die "MovieLens data not found in $ML_TEST_DIR"
fi

declare -a test_files=()
if (($#)); then
    msg -dbg "using tests from CLI"
    test_files=("$@")
else
    msg -dbg "scanning for tests"
    test_files=($TEST_DIR/test-*.sh)
fi

msg "running ${#test_files[@]} test suites"
declare -a taps=()
for test in "${test_files[@]}"; do
    start-group "CLI test $test"
    msg "running test $test"
    tap_file="${test%%.sh}.tap"
    msg -dbg "saving output to $tap_file"
    export TEST_WORK=$(mktemp -d)
    msg -dbg "using temporary directory $TEST_WORK"
    msg -dbg "invoking test"
    bash --noprofile --norc "$TEST_DIR/harness.sh" "$test" 5>"$tap_file"
    status="$?"
    if (($status)); then
        msg -err "test $test errored with $status"
    else
        msg "test $test completed"
    fi
    rm -rf "$TEST_WORK"
    if [[ -s "$tap_file" ]]; then
        taps+=("$tap_file")
    fi
    end-group
done

if [[ $usage_coverage = true ]]; then
    start-group "test coverage"
    coverage report
    end-group
fi

start-group "CLI test summary"
exec tappy "${taps[@]}"
end-group
