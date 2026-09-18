#!/usr/bin/env bash

. "$(dirname "$0")/lib/init.sh" || exit 2

_cli_definition() {
    setup TESTS help:_cli_help -- "Usage: scripts/test.sh [options] [TEST...]" ''
    msg -- "Options:"
    flag LOG_VERBOSE init:: -v --verbose -- "verbose log output"
    flag TEST_RELEASE --release -- "test release build of accelerator"
    flag SLOW_TESTS --slow -- "include slow tests"
    flag COVER --coverage -- "measure test coverage"
    flag ACCEL_COVER --accel-coverage -- "run tests with Rust accelerator coverage"
    disp :_cli_help -h --help -- "show help"
}
eval "$(getoptions _cli_definition _cli_parse "$@") exit 1"
_cli_parse "$@"
eval "set -- $TESTS"

. "${UV_PROJECT_ENVIRONMENT:-$PROJECT_ROOT/.venv}/bin/activate"

declare -a test_args=(--durations=10)
declare -a build_args=()

if (($COVER)); then
    msg "running tests with coverage"
    test_args+=(--cov=src/lenskit --cov-report=xml --cov-report=term)
fi

if (($TEST_RELEASE)); then
    build_args+=(-r)
fi

msg "building accelerator"
run-cmd -check just build-accel "${build_args[@]}"

if (($ACCEL_COVER)); then
    build_args+=(--coverage)
    export LLVM_PROFILE_FILE="$PWD/.coverage-prof/lenskit-test-%p-%m.profraw"
    just clean-rust-coverage || die "cannot clean coverage"
    msg "re-building accelerator with coverage"
    run-cmd -check maturin develop "${build_args[@]}" -- -C instrument-coverage
fi

if (($SLOW_TESTS)); then
    test_args+=(-m 'not slow')
fi

# need eval to properly quote these arguments
msg "running tests"
run-cmd -check pytest "${test_args[@]}" "$@"

if (($ACCEL_COVER)); then
    just collect-rust-coverage || die "failed to collect coverage data"
fi
