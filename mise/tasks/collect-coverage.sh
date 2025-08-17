#!/usr/bin/env bash
#MISE description="Collect code coverage"
#MISE wait_for=["test"]

set -eo pipefail

cargo profdata merge -sparse .coverage-prof/lenskit-tests-*.profraw -o .coverage-prof/lenskit-tests.profdata
cargo cov -- export --instr-profile=.coverage-prof/lenskit-tests.profdata target/debug/lib_accel.dylib --format=lcov >lcov.info
if [[ -e .coverage ]]; then
    echo-run coverage xml
fi
