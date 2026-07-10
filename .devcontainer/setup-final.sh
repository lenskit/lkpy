#!/bin/bash
set -xeo pipefail

# sync again to build LensKit post-attach
mise x -- uv sync --all-extras --group=cpu
