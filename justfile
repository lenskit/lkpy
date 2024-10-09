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

# set up for development in non-conda environments
install-dev:
    {{PIP}} install -r dev-requirements.txt -e {{ prepend('-e ', PACKAGES) }} --all-extras

# create a conda environment file for development
conda-dev-env-file *components=PACKAGES:
    pipx run --no-cache --spec . lk-conda -o environment.yml \
         -p {{python}} -e all requirements-dev.txt \
        pyproject.toml {{ append('/pyproject.toml', components) }}

# create a conda environment for development
conda-dev-env *components=PACKAGES: (conda-dev-env-file components)
    conda env update --prune -n lkpy -f environment.yml
    pip install --no-deps -e .
    for pkg in {{components}}; do \
        echo "installing $pkg"; \
        pip install --no-deps -e $pkg; \
    done

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
    sphinx-autobuild --watch lenskit docs build/doc

# update the environment file used to install documentation
update-doc-env:
    python -m lkdev.conda -o docs/environment.yml \
        -e all requirements-doc.txt docs/doc-dep-constraints.yml \
        {{ append('/pyproject.toml', PACKAGES) }}
    -pre-commit run --files docs/environment.yml

# update source file headers
update-headers:
    unbehead

# update GH workflows
update-workflows:
    deno check workflows/*.ts
    {{DENO}} workflows/render.ts --github -o .github/workflows/test.yaml workflows/test.ts
    {{DENO}} workflows/render.ts --github -o .github/workflows/docs.yaml workflows/docs.ts
    -pre-commit run --files .github/workflows/*.yml
