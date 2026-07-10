# Python recommendation tools

[![Automatic Tests](https://github.com/lenskit/lkpy/actions/workflows/test.yml/badge.svg)](https://github.com/lenskit/lkpy/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/lenskit/lkpy/graph/badge.svg?token=DaGn7NFM2P)](https://codecov.io/gh/lenskit/lkpy)
[![Scientific Python Ecosystem Coordination](https://img.shields.io/badge/SPEC-0,1,7-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/)
[![PyPI - Version](https://img.shields.io/pypi/v/lenskit)](https://pypi.org/project/lenskit)
![Conda Version](https://img.shields.io/conda/vn/conda-forge/lenskit)

LensKit is a set of Python tools for experimenting with and studying recommender
systems. It provides support for training, running, and evaluating recommender
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

> [!NOTE]
>
> LensKit had significant changes in the 2025.1 release. See the [Migration
> Guide](https://lkpy.lenskit.org/stable/guide/migrating.html) for details.

[release]: https://lkpy.lenskit.org/en/stable/

## Installing

To install the current release with `uv` (recommended):

```console
$ uv pip install lenskit
```

Or, to add it to your project's dependencies and virtual environment:

```console
$ uv add lenskit
```

Classic `pip` also works:

```console
$ python -m pip install lenskit
```

Then see [Getting Started](https://lkpy.lenskit.org/stable/guide/GettingStarted.html)

### Conda Packages

You can also install LensKit from `conda-forge` with `pixi`:

```console
$ pixi add lenskit
```

Or `conda`:

```console
$ conda install -c conda-forge lenskit
```

### Development Version

To use the latest development version, you have two options. You can install
directly from GitHub:

```console
$ uv pip install -U git+https://github.com/lenskit/lkpy
```

Or you can use our PyPI index, by adding to `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "lenskit"
url = "https://pypi.lenskit.org/lenskit-dev/"
```

Binary wheels of LensKit development (and release) versions are automatically
pushed to this index, although they are not guaranteed to be permanently
available. Reproducible code should generally depend on released versions
published to PyPI.

### Simplifying PyTorch installation

We also provide mirrors of the PyTorch package repositories that are filtered to
only include PyTorch and directly supporting dependencies, without other
packages that conflict with or mask packages from PyPI, and with fallbacks for
other platforms (i.e., our CUDA indices include CPU-only MacOS packages). This
makes it easier to install specific versions of PyTorch in your project with
the index priority and fallthrough logic implemented by `uv`. To make your
project only use CPU-based PyTorch, you can add to `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "torch-cpu"
url = "https://pypi.lenskit.org/torch/cpu/"
```

Or CUDA 12.8:

```toml
[[tool.uv.index]]
name = "torch-gpu"
url = "https://pypi.lenskit.org/torch/cu128/"
```

These indices provide the same package distributions as the official PyTorch
repositories (in fact, they link directly to the PyTorch packages). They are
just an alternate index view that reduces some package conflicts.

## Developing

[issues]: https://github.com/lenskit/lkpy/issues
[workflow]: https://github.com/lenskit/lkpy/wiki/DevWorkflow
[Mise]: https://mise.jdx.dev
[hk]: https://hk.jdx.dev

To contribute to LensKit, clone or fork the repository, get to work, and submit
a pull request. We welcome contributions from anyone; if you are looking for a
place to get started, see the [issue tracker][issues].

Our development workflow is documented in [the wiki][workflow]; the wiki also
contains other information on _developing_ LensKit. User-facing documentation is
at <https://lenskit.org>.

We use [`uv`](https://astral.sh/uv/) for developing LensKit and managing
development environments. Our `pyproject.toml` file contains the Python
development dependencies; you also need a working Rust compiler (typically via
[`rustup`](https://rustup.rs/)). We provide [Mise][] configuration to
automatically install everything needed, including `uv` and `rust`.

The easiest way to work on LensKit is to **use the devcontainer** — in Visual
Studio Code, Zed, and other editors supporting Dev Containers, just re-open the
project in a dev container, and the necessary software will be automatically
installed.

> [!NOTE]
>
> The dev container is the only supported way to develop on Windows — while
> LensKit works and is regularly tested on Windows, we have not invested time
> in making sure the development environment works on Windows.

If you want to set up yourself, we recommend using [Mise][]:

```console
$ mise trust
$ mise install
$ uv sync
```

`mise install` will automatically install `uv`, `rust`, development support tools
and the Git pre-commit hooks (managed with [hk][]). You will also need a working
C compiler (on macOS, install Xcode or the Xcode command-line tools).

If you want to use a specific Python version, select it with `uv venv` or `uv sync`:

```console
$ uv venv -p 3.14t
$ uv sync
```

If you want all extras, do:

```console
$ uv sync --all-extras
```

## Testing Changes

You should always test your changes by running the LensKit test suite:

    uv run pytest tests

If you want to use your changes in a LensKit experiment, you can locally install
your modified LensKit into your experiment's environment. We recommend using
separate environments for LensKit development and for each experiment; you will
need to install the modified LensKit into your experiment's repository:

    uv pip install -e /path/to/lkpy

## Resources

- [Documentation](https://lkpy.lenskit.org)
- [Discussion and Announcements](https://github.com/orgs/lenskit/discussions)

## Acknowledgements

This material is based upon work supported by the National Science Foundation
under Grant No. IIS 17-51278. Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the author(s) and do not
necessarily reflect the views of the National Science Foundation.
