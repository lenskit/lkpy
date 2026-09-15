#!/bin/zsh
#MISE description="Run data tests in CI"
#MISE depends=["ci:prepare"]
#USAGE flag "-v --verbose" help="Enable verbose logging."

. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

step "running basic tests"
echo-run mise run test -- -v --coverage -m realdata
if (($?)); then
    die "tests failed"
fi

echo-run mise run test-cli --coverage --cov-append tests/cli/test-data-convert.sh
if (($?)); then
    die "CLI tests failed"
fi

step "uploading coverage"
mise run coverage:export || die "coverage export failed"
