#!/bin/sh

set -e
if [ ! -x "$HOME/miniconda/bin/conda" ]; then
    rm -rf "$HOME/miniconda"
    wget --no-verbose https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
fi
export PATH="$HOME/miniconda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
echo "updating conda"
conda update -q conda
if [ ! -d "$HOME/miniconda/envs/lkpy-test" ]; then
    echo "creating environment"
    conda create -q -n lkpy-test python="$TRAVIS_PYTHON_VERSION"
else
    echo "updating LKPY test environment"
    conda update -q -n lkpy-test --all
fi
echo 'installing scientific python'
conda install -q -n lkpy-test pandas scipy cython
echo "installing test utilities"
conda install -q -n lkpy-test pytest pytest-arraydiff pytest-xdist flake8 pylint invoke
if [ -n "$1" ]; then
    echo "installing extra utilities"
    conda install -q -n lkpy-test "$@"
fi
