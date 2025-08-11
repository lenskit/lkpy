#!/usr/bin/env bash
#MISE description="Build accelerator for in-place testing."
#MISE depends=["init-dirs"]

set -eo pipefail

. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

declare -A options

while getopts rh arg; do
    options[$arg]="${OPTARG:-1}"
done

if ((${options[h]})); then
    cat <<EOD
mise run build:accel [-h] [-P profile]

Options:
    -r      build in release mode
    -h      print this help message
EOD
    exit
fi

declare -a build_args=()
if ((${options[r]})); then
    build_args+='--profile=release'
else
    build_args+='--profile=dev'
fi

echo-run maturin develop "${build_args[@]}"
