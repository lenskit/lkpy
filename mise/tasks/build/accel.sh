#!/usr/bin/env bash
#MISE description="Build accelerator for in-place testing."
#MISE depends=["init-dirs"]
#USAGE flag "-r --release" help="build in release mode"
#USAGE flag "--coverage" help="build for code coverage"

set -eo pipefail

. "$MISE_PROJECT_ROOT/mise/task-functions.sh"
. "$MISE_PROJECT_ROOT/.venv/bin/activate"

declare -a build_args=()
if [[ $usage_release = true ]]; then
    build_args+='--profile=release'
else
    build_args+='--profile=dev'
fi

echo-run maturin develop "${build_args[@]}"

if [[ $usage_coverage = true ]]; then
    msg "rebuilding final stage for coverage"
    echo-run maturin develop "${build_args[@]}" -- -C instrument-coverage
fi
