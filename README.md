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
> DOI:[10.1145/3340531.3412778](https://dx.doi.org/10.1145/3340531.3412778>).
> arXiv:[1809.03125](https://arxiv.org/abs/1809.03125) [cs.IR].

## Installing

To install the current release with Anaconda (recommended):

    conda install -c lenskit lenskit

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


We recommend using an Anaconda environment for developing LensKit.  To set this
up, run:

    python setup.py dep_info --conda-environment dev-env.yml
    conda env create -f dev-env.yml

This will create a Conda environment called `lkpy-dev` with the packages
required to develop and test LensKit.

We don't maintain the Conda environment specification directly - instead, we
maintain information in `setup.cfg` to be able to generate it, so that we define
dependencies and versions in one place (well, two, if you count the `meta.yaml`
file used to build the Conda recipes).  The `dep_info` setuptools command will
generate a Conda environment specification from the current dependencies in
`setup.cfg`.

## Resources

- [Documentation](https://lkpy.lenskit.org)
- [Mailing list, etc.](https://lenskit.org/connect)

## Acknowledgements

This material is based upon work supported by the National Science Foundation
under Grant No. IIS 17-51278. Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the author(s) and do not
necessarily reflect the views of the National Science Foundation.
