# list the tasks in this project (default)
list-tasks:
    just --list

# clean up build artifacts
clean:
    rm -rf build dist *.egg-info
    rm -rvf lenskit/**/*.{c,so,dll}

# build the modules and wheels
build:
    python -m build -n

# build the extension modules in-place for testing
build-inplace:
    python setup.py build_ext --inplace

# install the package
[confirm("this installs package from a wheel, continue [y/N]?")]
install:
    pip install .

# install the package (editable)
install-editable:
    pip install -e .

# set up for development in non-conda environments
install-dev:
    pip install -e '.[dev,test,doc,sklearn]'

# set up a conda environment for development
setup-conda-env version="3.11" env="dev":
    conda env create -n lkpy -f envs/lenskit-py{{version}}-{{env}}.yaml

# run tests with default configuration
test: build-inplace
    python -m pytest

# run fast tests
test-fast: build-inplace
    python -m pytest -m 'not slow'

# run tests matching a keyword query
test-matching query: build-inplace
    python -m pytest -k {{query}}

# build documentation
docs:
    sphinx-build docs build/doc

# preview documentation with live rebuild
preview-docs:
    sphinx-autobuild --watch lenskit docs build/doc

# update environment specifications
update-envs:
    pyproject2conda project

# update source file headers
update-headers:
    unbehead
