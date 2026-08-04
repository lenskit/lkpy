#!/bin/zsh
#MISE description="Prepare CI environment"
#USAGE flag "-v --verbose" help="Enable verbose logging."

. "$MISE_PROJECT_ROOT/mise/task-functions.sh"
. "${UV_PROJECT_ENVIRONMENT:-$MISE_PROJECT_ROOT/.venv}/bin/activate"

if [[ -z $CI ]]; then
    msg "not in CI, skipping environment setup"
fi

if [[ $CI_SYSTEM_NAME = woodpecker ]]; then
    if [[ -d /datasets ]]; then
        msg "linking data from /datasets"
        for f in /datasets/*; do
            ln -s $f data/${f#/datasets/}
        done
    else
        msg -warn "running on Woodpecker but /datasets does not exist"
    fi
else
    msg "nothing to do for CI $CI_SYSTEM_NAME"
fi
