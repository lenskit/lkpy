#!/usr/bin/env bash

. "$(dirname "$0")/lib/init.sh" || exit 2

set -eo pipefail

tdir="target/debug"

OS="$(uname)"
if [[ $OS = Darwin ]]; then
    target="$tdir/liblenskit_accel.dylib"
else
    target="$tdir/liblenskit_accel.so"
fi

msg "merging coverage data"
run-cmd -check cargo profdata -- merge -sparse .coverage-prof/lenskit-test-*.profraw -o .coverage-prof/lenskit-test.profdata
msg "exporting lcov.info"
cargo cov -- export --instr-profile=.coverage-prof/lenskit-test.profdata "$target" --format=lcov >lcov.info
