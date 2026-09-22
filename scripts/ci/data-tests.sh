#!/bin/zsh

. "$(dirname "$0")/../lib/init.sh" || exit 2

msg -step "installing Python environment"
run-cmd -check uv venv -p 3.14 --clear
. .venv/bin/activate || die "cannot activate virtualenv"
run-cmd -check uv sync

msg -step "running basic tests"
run-cmd just test --coverage -- -m realdata
if (($?)); then
    die "tests failed"
fi

run-cmd just test-cli --coverage --cov-append tests/cli/test-data-convert.sh
if (($?)); then
    die "CLI tests failed"
fi

msg -step "exporting coverage"
run-cmd -check coverage xml
