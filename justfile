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

# Build the source code distribution
[group("build")]
[positional-arguments]
build-dist *ARGS='':
    ./scripts/build-dist.sh {{ ARGS }}

# Build the accelerator module
[arg('profile', long="release", short='r', value='release')]
[group("build")]
build-accel profile='dev':
    maturin develop --profile={{ profile }}

[group("build")]
[script]
build-conda: (build-dist '-dsc')
    export LK_PACKAGE_VERSION="$(./scripts/version-tool.py -q)"
    flags=
    if [ -n "${CI:-}" ]; then
        flags='--noarch-build-platform linux-64'
    fi
    set -x
    rattler-build build --recipe conda --output-dir dist/conda $flags

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

# build documentation site
[group("docs")]
build-docs: && build-schemas
    sphinx-build docs build/site

# preview and auto-build documentation site
[group("docs")]
preview-docs:
    sphinx-autobuild docs build/site

# build JSON schemas
[group("docs")]
build-schemas:
    mkdir -p build/site/schemas
    python -m lenskit.schemas -o build/site/schemas/config.json --config
    python -m lenskit.schemas -o build/site/schemas/pipeline.json --pipeline
    python -m lenskit.schemas -o build/site/schemas/tuner.json --tuner
