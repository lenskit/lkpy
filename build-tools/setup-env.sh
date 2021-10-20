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
        *) err "Invalid platform $1";;
    esac
    echo "::set-output name=conda_platform::$CONDA_PLATFORM"
    msg "Running with Conda platform $CONDA_PLATFORM"
}

extras=""
spec_opts=""

while getopts "p:V:e:s:" opt; do
    case $opt in
        p) os_plat="$OPTARG";;
        V) ptag=py$(echo "$OPTARG" | sed -e 's/\.//');;
        e) extras="$extras,$OPTARG";;
        s) spec_opts="$spec_opts -f build-tools/$OPTARG-spec.yml";;
        \?) err "invalid argument";;
    esac
done

if [ -n "$ptag" ]; then
    msg "Using Python tag $ptag"
else
    msg "No Python tag specified"
fi
# 2 cases: extras is empty, or it's not and has a leading comma
extras="$ptag$extras"

if [ -z "$os_plat" ]; then
    PLAT=$(uname |tr [A-Z] [a-z])
fi
set_platform "$os_plat"
test -n "$CONDA_PLATFORM" || err "conda platform not set for some reason"
msg "Installing Conda management tools"
vr conda install -qy -c conda-forge mamba conda-lock

msg "Preparing Conda environment lockfile"
vr conda-lock lock --mamba -k env -p $CONDA_PLATFORM -e "$extras" -f pyproject.toml $spec_opts

msg "Updating environment with Conda dependencies"
vr mamba env update -n base -f conda-$CONDA_PLATFORM.lock.yml
