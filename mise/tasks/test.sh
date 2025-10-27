#!/usr/bin/env bash
#MISE description="Run tests"
#USAGE flag "--coverage" help="Run tests with coverage."
#USAGE flag "--accel-coverage" help="Run tests with Rust accelerator coverage."
#USAGE flag "--slow" help="Include slow tests."
#USAGE flag "--release" help="Test the release-build accelerator."
#USAGE arg "<arg>" var=#true default="tests" help="Arguments and paths to pytest"

set -eo pipefail
. "$MISE_PROJECT_ROOT/mise/task-functions.sh"
. "$MISE_PROJECT_ROOT/.venv/bin/activate"

declare -a test_args=()
declare -a build_args=()

if [[ $usage_coverage = true ]]; then
    msg "running tests with coverage"
    test_args+=(--cov=src/lenskit --cov-report=xml --cov-report=term)
fi

if [[ $usage_release = true ]]; then
    build_args+=(-r)
fi

if [[ $usage_accel_coverage = true ]]; then
    build_args+=(--coverage)
    export LLVM_PROFILE_FILE="$PWD/.coverage-prof/lenskit-test-%p-%m.profraw"
    mise run coverage:clean-rust
fi

msg "building accelerator"
mise run build:accel -- "${build_args[@]}"

if [[ $usage_slow != true ]]; then
    test_args+=(-m 'not slow')
fi

# need eval to properly quote these arguments
eval "test_args+=($usage_arg)"
msg "running tests"
echo-run pytest "${test_args[@]}"

if [[ $usage_accel_coverage = true ]]; then
    mise run coverage:collect-rust
fi
