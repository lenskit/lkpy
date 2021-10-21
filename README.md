# Python recommendation tools

![Test Suite](https://github.com/lenskit/lkpy/workflows/Test%20Suite/badge.svg)
[![codecov](https://codecov.io/gh/lenskit/lkpy/branch/master/graph/badge.svg)](https://codecov.io/gh/lenskit/lkpy)
[![Maintainability](https://api.codeclimate.com/v1/badges/c02098c161112e19c148/maintainability)](https://codeclimate.com/github/lenskit/lkpy/maintainability)

LensKit is a set of Python tools for experimenting with and studying recommender
systems.  It provides support for training, running, and evaluating recommender
algorithms in a flexible fashion suitable for research and education.

LensKit for Python (LKPY) is the successor to the Java-based LensKit project.

If you use LensKit for Python in published research, please cite:

> Michael D. Ekstrand. 2020.
> LensKit for Python: Next-Generation Software for Recommender Systems Experiments.
> In <cite>Proceedings of the 29th ACM International Conference on Information and Knowledge Management</cite> (CIKM '20).
> DOI:[10.1145/3340531.3412778](https://dx.doi.org/10.1145/3340531.3412778).
> arXiv:[1809.03125](https://arxiv.org/abs/1809.03125) [cs.IR].

## Installing

To install the current release with Anaconda (recommended):

    conda install -c conda-forge lenskit

Or you can use `pip`:

    pip install lenskit

To use the latest development version, install directly from GitHub:

    pip install -U git+https://github.com/lenskit/lkpy

Then see [Getting Started](https://lkpy.lenskit.org/en/latest/GettingStarted.html)

## Developing

[issues]: https://github.com/lenskit/lkpy/issues
[workflow]: https://github.com/lenskit/lkpy/wiki/DevWorkflow

To contribute to LensKit, clone or fork the repository, get to work, and submit
a pull request.  We welcome contributions from anyone; if you are looking for a
place to get started, see the [issue tracker][].

Our development workflow is documented in [the wiki][workflow]; the wiki also
contains other information on *developing* LensKit. User-facing documentation is
at <https://lkpy.lenskit.org>.

[conda-lock]: https://github.com/conda-incubator/conda-lock

We recommend using an Anaconda environment for developing LensKit.
We don't maintain the Conda environment specification directly - instead, we
maintain information in `setup.toml` to be able to generate it, so that we define
dependencies and versions in one place.

[conda-lock][] can help you set up the environment (replace `linux-64` with your platform):

    # install conda-lock in base environment
    # alternatively: pip install conda-lock
    conda install -c conda-forge conda-lock
    # create the lock file for Python 3.9
    conda-lock -p linux-64 -f pyproject.toml -f lkbuild/python-3.9-spec.yml
    # create the environment
    conda env create -n lkpy -f conda-linux-64.lock

This will create a Conda environment called `lkpy` with the packages required to develop and test
LensKit.

We also provide support for automating some of this process through LensKit's
infrastructure utilities:

    invoke dev-lock

The `lkbuild/boot-env.yml` file defines a Conda environment with the packages necessary
for the lockfile generation to work.  The full set of commands:

    conda env create -f lkbuild/boot-env.yml
    conda activate lkboot
    invoke dev-lock
    conda create -n lkpy -f conda-linux-64.lock
    conda activate lkpy

`invoke dev-lock` can optionally specify the BLAS implementation (`openblas` or `mkl`), and the
Python version.

## Testing Changes

You should always test your changes by running the LensKit test suite:

    python -m pytest

If you want to use your changes in a LensKit experiment, you can locally install
your modified LensKit into your experiment's environment.  We recommend using
separate environments for LensKit development and for each experiment; you will
need to install the modified LensKit into your experiment's repository:

    conda activate my-exp
    conda install -c conda-forge flit
    cd /path/to/lkpy
    flit install --pth-file --deps none

You may need to first uninstall LensKit from your experiment repo; make sure that
LensKit's dependencies are all still installed.

Once you have pushed your code to a GitHub branch, you can use a Git repository as
a Pip dependency in an `environment.yml` for your experiment, to keep using the
correct modified version of LensKit until your changes make it in to a release.

## Resources

- [Documentation](https://lkpy.lenskit.org)
- [Mailing list, etc.](https://lenskit.org/connect)

## Acknowledgements

This material is based upon work supported by the National Science Foundation
under Grant No. IIS 17-51278. Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the author(s) and do not
necessarily reflect the views of the National Science Foundation.
