#!/usr/bin/env bash
#MISE description="Run tests"
#USAGE flag "--coverage" help="Run tests with coverage."
#USAGE flag "--slow" help="Include slow tests."
#USAGE flag "--release" help="Test the release-build accelerator."
#USAGE arg "<arg>" var=#true default="tests" help="Arguments and paths to pytest"

set -eo pipefail
source "$MISE_PROJECT_ROOT/mise/task-functions.sh"

declare -a test_args=()
declare -a build_args=()

if [[ $usage_coverage = true ]]; then
    msg "running tests with coverage"
    test_args+=(--cov=src/lenskit --cov-report=xml --cov-report=term)
fi

if [[ $usage_release = true ]]; then
    build_args+=(-r)
fi

msg "building accelerator"
mise run build:accel -- "${build_args[@]}"

if [[ $usage_slow != true ]]; then
    test_args+=(-m 'not slow')
fi

# need eval to properly quote these arguments
eval "test_args+=($usage_arg)"
echo-run pytest "${test_args[@]}"
