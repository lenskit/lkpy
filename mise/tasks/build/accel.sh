#!/usr/bin/env bash
#MISE description="Build accelerator for in-place testing."
#MISE depends=["init-dirs"]
#USAGE flag "-r --release" help="build in release mode"

set -eo pipefail

. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

declare -a build_args=()
if [[ $usage_release = true ]]; then
    build_args+='--profile=release'
else
    build_args+='--profile=dev'
fi

echo-run maturin develop "${build_args[@]}"
