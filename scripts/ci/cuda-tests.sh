#!/bin/zsh
#MISE description="Run data tests in CI"
#MISE depends=["ci:prepare"]
#USAGE flag "-v --verbose" help="Enable verbose logging."

. "$(dirname "$0")/../lib/init.sh" || exit 2

msg -step "installing Python environment"
run-cmd -check uv venv -p 3.14 --clear
. .venv/bin/activate || die "cannot activate virtualenv"
run-cmd -check uv sync --group=gpu

msg -step "checking LensKit install"
run-cmd -check lenskit doctor

msg -step "running test suite"
export LK_TORCH_COMPILE=0
run-cmd python -m pytest --verbose --durations=25 --cov=src/lenskit -m 'not realdata and not compat' tests
if (($?)); then
    die "tests failed"
fi

msg -step "exporting coverage"
run-cmd -check coverage xml
