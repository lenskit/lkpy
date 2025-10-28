#!/usr/bin/env bash
#MISE description="Collect code coverage"
#MISE wait_for=["test"]
#USAGE flag "-r --release" help="Profile with release data."

set -eo pipefail
. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

if [[ $usage_release = true ]]; then
    tdir="target/release"
else
    tdir="target/debug"
fi

OS="$(uname)"
if [[ $OS = Darwin ]]; then
    target="$tdir/lib_accel.dylib"
else
    target="$tdir/lib_accel.so"
fi

msg "merging coverage data"
cargo profdata -- merge -sparse .coverage-prof/lenskit-test-*.profraw -o .coverage-prof/lenskit-test.profdata
msg "exporting lcov.info"
cargo cov -- export --instr-profile=.coverage-prof/lenskit-test.profdata "$target" --format=lcov >lcov.info
