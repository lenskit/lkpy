#!/bin/bash

# Script to bootstrap an environment for running the infra code.
# The resulting environment can be used to run 'invoke' and get a
# Conda lock for a full development environment.
#
# This script is intended to be sourced, but can be run on its own.

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
  local ec
  echo + "$@" >&2
  "$@"
  ec="$?"
  if [ $ec -ne 0 ]; then
      echo "command $1 terminated with code $ec" >&2
      exit 2
  fi
}

setup_micromamba()
{
  msg "Installing Micromamba"
  CONDA_PLATFORM=$(python3 lkbuild/env.py)
  test "$?" -eq 0 || exit 2
  mkdir -p build/mmboot
  vr wget -qO build/micromamba.tar.bz2 https://micromamba.snakepit.net/api/micromamba/$CONDA_PLATFORM/latest
  vr tar -C build/mmboot -xvjf build/micromamba.tar.bz2
  MM=build/mmboot/bin/micromamba
  if [ ! -e "$MM" ]; then
    # this should only happen on Windows
    MM=build/mmboot/Library/bin/micromamba.exe
  fi
  eval "$($MM shell hook -p $HOME/micromamba -s bash)"
}

setup_boot_env()
{
  msg "Installing bootstrap environment"
  vr micromamba env create -qy -n lkboot -f lkbuild/boot-env.yml
  msg "Activating bootstrap environment"
  micromamba activate lkboot
}

setup_micromamba
setup_boot_env

msg "Inspecting environment"
CONDA_PLATFORM=$(invoke conda-platform)
