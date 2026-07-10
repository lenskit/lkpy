#!/bin/bash
set -xeo pipefail

sudo chown vscode:vscode target || true
# fix git permissions warning
git config --global --add safe.directory $PWD

# install dev tools
mise trust -a
mise install

# install the development environment
mise x -- uv venv -p 3.12
mise x -- uv sync --all-extras --group=cpu
