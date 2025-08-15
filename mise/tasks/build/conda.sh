#!/usr/bin/env bash
#MISE description="Build Conda package"
#MISE depends=["build:dist -sd"]

set -eo pipefail

. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

export LK_PACKAGE_VERSION="$(python "${MISE_PROJECT_ROOT}/mise/tasks/version.py" -q)"
export PYTHONPATH=

declare -a flags=()

if [[ -n "$CI" ]]; then
    flags+=(--noarch-build-platform linux-64)
fi

echo-run rattler-build --recipe conda --ouptut-dir dist/conda "${flags[@]}"
