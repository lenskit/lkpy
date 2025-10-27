#!/usr/bin/env bash
#MISE description="Collect code coverage"
#MISE wait_for=["test"]

set -eo pipefail
. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

msg "merging coverage data"
cargo profdata -- merge -sparse .coverage-prof/lenskit-test-*.profraw -o .coverage-prof/lenskit-test.profdata
msg "exporting lcov.info"
cargo cov -- export --instr-profile=.coverage-prof/lenskit-test.profdata target/release/lib_accel.dylib --format=lcov >lcov.info
