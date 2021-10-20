#!/bin/bash

msg()
{
    echo "$@" >&2
}

err()
{
    echo "ERR:" "$@" >&2
    exit 3
}

vr()
{
    echo + "$@" >&2
    "$@"
    if [ $? -ne 0 ]; then
        echo "command $1 terminated with code $?" >&2
        exit 2
    fi
}

set_platform()
{
    case "$1" in
        osx-*|linux-*|win-*) CONDA_PLATFORM="$1";;
        ubuntu) CONDA_PLATFORM=linux-64;;
        macos) CONDA_PLATFORM=osx-64;;
        windows) CONDA_PLATFORM=win-64;;
        *) err "Invalid platform";;
    esac
    echo "::set-output name=conda_platform::$CONDA_PLATFORM"
    msg "Running with Conda platform $CONDA_PLATFORM"
}

extras=""

while getopts "p:V:e:" opt; do
    case $opt in
        p) os_plat="$OPTARG";;
        V) ptag=py$(echo "$OPTARG" | sed -e 's/\.//');;
        e) extras="$extras,$OPTARG";;
        \?) err "invalid argument";;
    esac
done

msg "Using Python tag $pytag"
# 2 cases: extras is empty, or it's not and has a leading comma
extras="$ptag$extras"

scan_platform "$PLAT"
msg "Installing Conda management tools"
vr conda install -qy -c conda-forge mamba conda-lock

msg "Preparing Conda environment lockfile"
vr conda-lock --mamba -k env -p $CONDA_PLATFORM -e "$extras" -f pyproject.toml

msg "Updating environment with Conda dependencies"
vr mamba env update -n base conda-$CONDA_PLATFORM.lock.yml
