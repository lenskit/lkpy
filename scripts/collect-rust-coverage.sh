#!/usr/bin/env bash

. "$(dirname "$0")/lib/init.sh" || exit 2

set -o pipefail

tdir="target/debug"

OS="$(uname)"
if [[ $OS = Darwin ]]; then
    target="$tdir/liblenskit_accel.dylib"
else
    target="$tdir/liblenskit_accel.so"
fi
sysroot="$(rustc --print sysroot)"
if (($?)); then die "cannot find rust sysroot"; fi
tuple="$(rustc --print host-tuple)"
if (($?)); then die "cannot find rust host tuple"; fi

PATH="$sysroot/lib/rustlib/$tuple/bin:$PATH"

msg "merging coverage data"
run-cmd -check llvm-profdata merge --sparse .coverage-prof/lenskit-test-*.profraw -o .coverage-prof/lenskit-test.profdata
msg "exporting lcov.info"
run-cmd -check llvm-cov export --instr-profile=.coverage-prof/lenskit-test.profdata "$target" --format=lcov >lcov.info
