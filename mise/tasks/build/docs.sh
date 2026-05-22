#!/bin/bash
#MISE description="Build documentation."
#USAGE flag "-P --preview" help="Run preview server."

cmd=(sphinx-build)
. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

if [[ $usage_preview ]]; then
    cmd=(sphinx-autobuild)
fi

echo-run "${cmd[@]}" docs build/site || exit 2
