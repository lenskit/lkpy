#!/usr/bin/env -S usage bash
#USAGE flag "-v --verbose" help="Output verbose log messages."
#USAGE flag "--coverage" help="Run with test coverage."
#USAGE flag "--cov-append" help="Append to existing test coverage."
#USAGE arg "[test]" var=#true help="Test files to run."

. "$PWD/mise/task-functions.sh" || exit 2
export TEST_DIR="$(dirname "$0")"

export VERBOSE

PYRUN="python"
if [[ $usage_coverage = true ]]; then
    PYRUN="coverage run -a"

    if [[ $usage_cov_append = true ]]; then
        msg "appending to test coverage"
    else
        msg "resetting test coverage"
        coverage erase || exit 2
    fi
fi
export PYRUN

export ML_TEST_DIR="data/ml-latest-small"
if [[ ! -f "$ML_TEST_DIR/ratings.csv" ]]; then
    die "MovieLens data not found in $ML_TEST_DIR"
fi

declare -a test_files=()
if [[ $usage_test ]]; then
    dbg "using tests from CLI"
    test_files=($usage_test)
else
    dbg "scanning for tests"
    test_files=($TEST_DIR/test-*.sh)
fi

msg "running ${#test_files[@]} test suites"
declare -a taps=()
for test in "${test_files[@]}"; do
    msg "running test $test"
    tap_file="${test%%.sh}.tap"
    dbg "saving output to $tap_file"
    export TEST_WORK=$(mktemp -d)
    dbg "using temporary directory $TEST_WORK"
    dbg "invoking test"
    bash --noprofile --norc "$TEST_DIR/harness.sh" "$test" 5>"$tap_file"
    status="$?"
    if (($status)); then
        err "test $test errored with $status"
    else
        msg "test $test completed"
    fi
    rm -rf "$TEST_WORK"
    taps+=("$tap_file")
done

if [[ $usage_coverage = true ]]; then
    coverage report
fi

exec prove "${taps[@]}"
