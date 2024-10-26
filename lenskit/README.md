# Python recommendation tools

![Test Suite](https://github.com/lenskit/lkpy/workflows/Test%20Suite/badge.svg)
[![codecov](https://codecov.io/gh/lenskit/lkpy/branch/master/graph/badge.svg)](https://codecov.io/gh/lenskit/lkpy)

LensKit is a set of Python tools for experimenting with and studying recommender
systems.  It provides support for training, running, and evaluating recommender
algorithms in a flexible fashion suitable for research and education.

LensKit for Python (LKPY) is the successor to the Java-based LensKit project.

> [!IMPORTANT]
> If you use LensKit for Python in published research, please cite:
>
> > Michael D. Ekstrand. 2020.
> > LensKit for Python: Next-Generation Software for Recommender Systems Experiments.
> > In <cite>Proceedings of the 29th ACM International Conference on Information and Knowledge Management</cite> (CIKM '20).
> > DOI:[10.1145/3340531.3412778](https://dx.doi.org/10.1145/3340531.3412778).
> > arXiv:[1809.03125](https://arxiv.org/abs/1809.03125) [cs.IR].

> [!WARNING]
> This is the `main` branch of LensKit, following new development in preparation
> for the 2024 release.  It will be changing frequently and incompatibly. You
> probably want to use a [stable release][release].

[release]: https://lkpy.lenskit.org/en/stable/

## Installing

To install the current release with Anaconda (recommended):

    conda install -c conda-forge lenskit

Or you can use `pip`:

    pip install lenskit

To use the latest development version, install directly from GitHub:

    pip install -U git+https://github.com/lenskit/lkpy

Then see [Getting Started](https://lkpy.lenskit.org/en/latest/GettingStarted.html)

## Acknowledgements

This material is based upon work supported by the National Science Foundation
under Grant No. IIS 17-51278. Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the author(s) and do not
necessarily reflect the views of the National Science Foundation.
