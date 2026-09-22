#!/bin/zsh

. "$(dirname "$0")/../lib/init.sh" || exit 2

run-cmd -check uv venv -p 3.14 --clear
. .venv/bin/activate || die "cannot activate virtualenv"

step "installing Python environment"
run-cmd -check uv sync

step "running basic tests"
run-cmd just test -v --coverage -m realdata
if (($?)); then
    die "tests failed"
fi

run-cmd just test-cli --coverage --cov-append tests/cli/test-data-convert.sh
if (($?)); then
    die "CLI tests failed"
fi

step "exporting coverage"
run-cmd -check coverage xml
