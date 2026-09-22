#!/bin/zsh
#MISE description="Run data tests in CI"
#MISE depends=["ci:prepare"]
#USAGE flag "-v --verbose" help="Enable verbose logging."

. "$(dirname "$0")/../lib/init.sh" || exit 2

run-cmd -check uv venv -p 3.14 --clear
. .venv/bin/activate || die "cannot activate virtualenv"

step "installing Python environment"
run-cmd -check uv sync --group=gpu

step "checking LensKit install"
run-cmd -check uv run lenskit doctor

step "running test suite"
run-cmd just test -v --coverage -m 'not slow'
if (($?)); then
    die "tests failed"
fi

step "exporting coverage"
run-cmd -check coverage xml
