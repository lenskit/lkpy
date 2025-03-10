PIP := "uv pip"
python := "3.11"
conda_env := "lkpy"
DENO := "deno run --allow-read=. --allow-write=.github/workflows --allow-net=jsr.io"

# list the tasks in this project (default)
list-tasks:
    just --list

# clean up build artifacts
clean:
    rm -rf build dist *.egg-info */dist */build
    git clean -xf docs

# build the modules and wheels
build:
    python -m build -n -o dist

build-sdist:
    python -m build -n -s -o dist

# install the package
[confirm("this installs package from a wheel, continue [y/N]?")]
install:
    {{PIP}} install .

# install the package (editable)
install-editable:
    {{PIP}} install -e  .

# run tests with default configuration
test:
    python -m pytest

# run fast tests
test-fast:
    python -m pytest -m 'not slow'

# run tests matching a keyword query
test-matching query:
    python -m pytest -k "{{query}}"

# build documentation
docs:
    sphinx-build docs build/doc

# preview documentation with live rebuild
preview-docs:
    sphinx-autobuild --watch src docs build/doc

# update the BibTeX file (likely only works on Michael's laptop)
update-bibtex:
    curl -o docs/lenskit.bib 'http://127.0.0.1:23119/better-bibtex/export/collection?/4/9JMHQD9K.bibtex'

# update source file headers
update-headers:
    unbehead

# update GH workflows
update-workflows:
    deno check workflows/*.ts
    {{DENO}} workflows/render.ts --github -o .github/workflows/test.yml workflows/test.ts
    {{DENO}} workflows/render.ts --github -o .github/workflows/docs.yml workflows/docs.ts
    -pre-commit run --files .github/workflows/*.yml
