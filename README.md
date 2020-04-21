# Python recommendation tools

![Test Suite](https://github.com/lenskit/lkpy/workflows/Test%20Suite/badge.svg)
[![codecov](https://codecov.io/gh/lenskit/lkpy/branch/master/graph/badge.svg)](https://codecov.io/gh/lenskit/lkpy)
[![Maintainability](https://api.codeclimate.com/v1/badges/c02098c161112e19c148/maintainability)](https://codeclimate.com/github/lenskit/lkpy/maintainability)

LensKit is a set of Python tools for experimenting with and studying recommender
systems.  It provides support for training, running, and evaluating recommender
algorithms in a flexible fashion suitable for research and education.

Python LensKit (LKPY) is the successor to the Java-based LensKit project.

## Installing

To install the current release with Anaconda (recommended):

    conda install -c lenskit lenskit

Or you can use `pip`:

    pip install lenskit

To use the latest development version, install directly from GitHub:

    pip install -U git+https://github.com/lenskit/lkpy

Then see [Getting Started](https://lkpy.lenskit.org/en/latest/GettingStarted.html)

## Developing

To contribute to LensKit, clone or fork the repository, get to work, and submit a pull request.  We welcome contributions from anyone; if you are looking for a place to get started, see the [issue tracker](https://github.com/lenskit/lkpy/issues).

We recommend using an Anaconda environment for developing LensKit.  To set this up, run:

    conda env create -f dev-env-py37.yml

This will create a Conda environment called `lkpy-dev` with the packages required to develop and test LensKit.

## Resources

- [Documentation](https://lkpy.lenskit.org)
- [Mailing list, etc.](https://lenskit.org/connect)

## Acknowledgements

This material is based upon work supported by the National Science Foundation under Grant No. IIS 17-51278.
Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
