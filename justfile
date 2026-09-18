BBT_URL := "http://127.0.0.1:23119/better-bibtex/export?/group;name:LensKit/collection/LensKit%20References.bibtex"

[default]
_info:
    just -l

# update BibTeX file
update-bib:
    curl -fL -o docs/lenskit.bib "{{ BBT_URL }}"

# Print LensKit version information
version:
    ./scripts/version-tool.py

# Build the accelerator module
[arg('profile', long="release", short='r', value='release')]
build-accel profile='dev':
    maturin develop --profile={{ profile }}

# run the LensKit tests (see scripts/test.sh)
[positional-arguments]
test *ARGS='':
    ./scripts/test.sh "$@"

# clean up Rust code coverage (used prior to tests)
_clean-rust-coverage:
    rm -rf .coverage-prof
    mkdir .coverage-prof

# collect Rust code coverage into usable output
_collect-rust-coverage:
    ./scripts/collect-rust-coverage.sh
