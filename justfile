PIP := "uv pip"

# list the tasks in this project (default)
list-tasks:
    just --list

# clean up build artifacts
clean:
    rm -rf build dist *.egg-info
    # we don't have extension modules rn, but in case we do, remove them
    rm -rvf lenskit/**/*.{c,so,dll}

# build the modules and wheels
build:
    python -m build -n -o dist lenskit
    python -m build -n -o dist lenskit-funksvd

# install the package
[confirm("this installs package from a wheel, continue [y/N]?")]
install:
    {{PIP}} install .

# install the package (editable)
install-editable:
    {{PIP}} install -e .

# set up for development in non-conda environments
install-dev:
    {{PIP}} install -r dev-requirements.txt -e . --all-extras

# set up a conda environment for development
setup-conda-env version="3.11" env="dev":
    conda env create -n lkpy -f envs/lenskit-py{{version}}-{{env}}.yaml

# run tests with default configuration
test:
    python -m pytest

# run fast tests
test-fast:
    python -m pytest -m 'not slow'

# run tests matching a keyword query
test-matching query:
    python -m pytest -k {{query}}

# build documentation
docs:
    sphinx-build docs build/doc

# preview documentation with live rebuild
preview-docs:
    sphinx-autobuild --watch lenskit docs build/doc

# update source file headers
update-headers:
    unbehead

# update GH workflows
update-workflows:
    python ./utils/render-test-workflow.py -o .github/workflows/test.yml
