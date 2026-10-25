#!/bin/bash
set -xeo pipefail

# sync again to build LensKit post-attach
uv sync --all-extras --group=cpu
"$VIRTUAL_ENV/bin/prek" install --prepare-hooks
