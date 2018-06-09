#!/bin/sh

# INTENDED TO BE SOURCED
if [[ "$PYTHON_TYPE" = conda ]]; then
    export PATH="$HOME/miniconda/bin:$PATH"
    source activate lkpy-test
else
    echo >&2 "Conda not requested, skipping"
fi