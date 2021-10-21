#!/bin/sh

# Create a lock file for CI builds, and register with GitHub outputs.

. ./lkbuild/bootstrap-env.sh

vr invoke dev-lock "$@"
echo "::set-output name=environment-file::conda-$CONDA_PLATFORM.lock"
