#!/usr/bin/env bash
#MISE description="Run CLI tests"
#MISE wait_for=["test"]

exec ./tests/cli/run.sh "$@"
