CACHEDIR_TAG := "Signature: 8a477f597d28d172789f06886806bc55"
BIBTEX_URL := "http://127.0.0.1:23119/better-bibtex/export/collection?/4/9JMHQD9K.bibtex"

# list available recipes
default:
    @just -l

# intiialize output directories
init-dirs:
    #!/bin/bash
    set -euo pipefail
    for dir in build output; do
        mkdir -p $dir
        if [ ! -f $dir/CACHEDIR.TAG ]; then
            echo "{{CACHEDIR_TAG}}" >$dir/CACHEDIR.TAG
        fi
    done

# build the source distribution
build-sdist: init-dirs
    uv build --sdist

# build packages for the current platform
build-dist: init-dirs
    uv build

# build the accelerator in-place
build-accel profile="dev": init-dirs
    maturin develop --profile="{{profile}}"

# run the tests
test slow="yes" +args='tests': build-accel
    pytest {{ if slow == "yes" {""} else {"-m 'not slow'"} }} {{args}}

# build the documentation
docs: init-dirs
    sphinx-build docs build/doc

# auto-build and preview documentation
preview-docs: init-dirs
    sphinx-autobuild --watch src docs build/doc

# update the BibTeX file
update-bibtex:
    curl -fo docs/lenskit.bib "{{BIBTEX_URL}}"
