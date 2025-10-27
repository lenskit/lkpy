#!/usr/bin/env bash
#MISE description="Clean Rust coverage data."

set -eo pipefail
. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

echo-run rm -rf .coverage-prof
echo-run mkdir .coverage-prof
