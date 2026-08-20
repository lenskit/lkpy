#!/bin/zsh
#MISE description="Run data tests in CI"
#MISE depends=["ci:prepare"]
#USAGE flag "-v --verbose" help="Enable verbose logging."

. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

msg "installing Python deps"
echo-run uv sync -p 3.14 --group=gpu

echo-run uv run lenskit doctor

echo-run mise run test -- -v --coverage -m 'not slow'
if (($?)); then
    die "tests failed"
fi

mise run coverage:export || die "coverage export failed"
