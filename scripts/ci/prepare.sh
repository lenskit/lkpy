#!/bin/zsh

. "$(dirname "$0")/../lib/init.sh" || exit 2

msg -step "preparing CI environment"

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
fi
