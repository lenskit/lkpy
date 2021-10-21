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

gh_out()
{
  local name
  name="$1"
  shift
  echo "::set-output name=$name::$*"
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
    gh_out conda-platform "$CONDA_PLATFORM"
    msg "Running with Conda platform $CONDA_PLATFORM"
}

setup_micromamba()
{
  msg "Installing Micromamba"
  vr wget -qO build-tools/micromamba.tar.bz2 https://micromamba.snakepit.net/api/micromamba/$CONDA_PLATFORM/latest
  vr tar -C build-tools -xvjf build-tools/micromamba.tar.bz2
  if [ $CONDA_PLATFORM = win-64 ]; then
    MM=build-tools/Library/bin/micromamba.exe
  else
    MM=build-tools/bin/micromamba
  fi
  eval "$($MM shell hook -p $HOME/micromamba -s bash)"
}

setup_boot_env()
{
  msg "Installing bootstrap environment"
  vr micromamba env create -y -n lkboot -f build-tools/boot-env.yml
  msg "Activating bootstrap environment"
  micromamba activate lkboot
}

if [ ! -d build-tools ]; then
  err "build-tools not found, is this being run from package root?"
fi

extras=""
spec_opts=""
env_name="lktest"
mode=lock

while getopts "p:V:e:s:n:" opt; do
    case $opt in
        n) env_name="$OPTARG";;
        p) os_plat="$OPTARG";;
        V) spec_opts="$spec_opts -f build-tools/python-$OPTARG-spec.yml";;
        e) if [ -z "$extras" ]; then extras="$OPTARG"; else extras="$extras,$OPTARG"; fi;;
        s) spec_opts="$spec_opts -f build-tools/$OPTARG-spec.yml";;
        \?) err "invalid argument";;
    esac
done

if [ -z "$os_plat" ]; then
    PLAT=$(uname |tr [A-Z] [a-z])
fi
set_platform "$os_plat"
test -n "$CONDA_PLATFORM" || err "conda platform not set for some reason"

setup_micromamba
setup_boot_env

msg "Preparing Conda environment lockfile"
vr conda-lock lock --mamba -k env -p $CONDA_PLATFORM -e "$extras" -f pyproject.toml $spec_opts

gh_out environment-file "conda-$CONDA_PLATFORM.lock.yml"

if [ "$mode" = install ]; then
  msg "Creating Conda environment"
  vr mamba env create -n "$env_name" -f conda-$CONDA_PLATFORM.lock.yml
  gh_out environment-name "$env_name"
fi
