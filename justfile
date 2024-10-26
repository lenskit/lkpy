PIP := "uv pip"
PACKAGES := "lenskit lenskit-funksvd lenskit-implicit lenskit-hpf"
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
    for pkg in {{PACKAGES}}; do python -m build -n -o dist $pkg; done

build-sdist:
    for pkg in {{PACKAGES}}; do python -m build -n -s -o dist $pkg; done

# install the package
[confirm("this installs package from a wheel, continue [y/N]?")]
install:
    {{PIP}} install {{PACKAGES}}

# install the package (editable)
install-editable:
    {{PIP}} install {{ prepend('-e ', PACKAGES) }}

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
    sphinx-autobuild --watch lenskit/lenskit docs build/doc

# update source file headers
update-headers:
    unbehead

# update GH workflows
update-workflows:
    deno check workflows/*.ts
    {{DENO}} workflows/render.ts --github -o .github/workflows/test.yml workflows/test.ts
    {{DENO}} workflows/render.ts --github -o .github/workflows/docs.yml workflows/docs.ts
    -pre-commit run --files .github/workflows/*.yml
