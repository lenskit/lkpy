#!/bin/bash
set -xeo pipefail

sudo chown vscode:vscode target || true
# fix git permissions warning
git config --global --add safe.directory $PWD

# install the development environment
uv venv -p 3.12
uv sync --all-extras --group=cpu
