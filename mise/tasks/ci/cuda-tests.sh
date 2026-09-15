#!/bin/zsh
#MISE description="Run data tests in CI"
#MISE depends=["ci:prepare"]
#USAGE flag "-v --verbose" help="Enable verbose logging."

. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

step "installing Python environment"
echo-run uv sync -p 3.14 --group=gpu

step "checking LensKit install"
echo-run uv run lenskit doctor

step "running test suite"
echo-run mise run test -- -v --coverage -m 'not slow'
if (($?)); then
    die "tests failed"
fi

step "uploading coverage"
mise run coverage:export || die "coverage export failed"
