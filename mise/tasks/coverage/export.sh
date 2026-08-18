#!/usr/bin/env zsh
#MISE description="Export test coverage data."
#MISE wait_for=["test", "test-cli"]

coverage xml
