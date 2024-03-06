default:
    just --list

# clean up build artifacts
clean:
    rm -rf build dist *.egg-info
    rm -rvf lenskit/**/*.{c,so,dll}

# build the modules and wheels
build:
    python -m build -n

# install the package
[confirm("this installs package from a wheel, continue [y/N]?")]
install:
    pip install .

# install the package (editable)
install-editable:
    pip install -e .

# set up for development
install-dev:
    pip install -e '.[dev,test,doc,sklearn]'

# run tests with default configuration
test:
    python -m pytest

# run fast tests
test-fast:
    python -m pytest -m 'not slow'

# run tests matching a keyword query
test-matching query:
    python -m pytest -k {{query}}

# update environment specifications
update-envs:
    pyproject2conda project
