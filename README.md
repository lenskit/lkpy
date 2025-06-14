# Python recommendation tools

[![Automatic Tests](https://github.com/lenskit/lkpy/actions/workflows/test.yml/badge.svg)](https://github.com/lenskit/lkpy/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/lenskit/lkpy/graph/badge.svg?token=DaGn7NFM2P)](https://codecov.io/gh/lenskit/lkpy)
[![Scientific Python Ecosystem Coordination](https://img.shields.io/badge/SPEC-0,1,7-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/)
[![PyPI - Version](https://img.shields.io/pypi/v/lenskit)](https://pypi.org/project/lenskit)
![Conda Version](https://img.shields.io/conda/vn/conda-forge/lenskit)

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

> [!NOTE]
>
> LensKit had significant changes in the 2025.1 release.  See the [Migration
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
available.  Reproducible code should generally depend on released versions
published to PyPI.

### Simplifying PyTorch installation

We also provide mirrors of the PyTorch package repositories that are filtered to
only include PyTorch and directly supporting dependencies, without other
packages that conflict with or mask packages from PyPI, and with fallbacks for
other platforms (i.e., our CUDA indices include CPU-only MacOS packages).  This
makes it easier to install specific versions of PyTorch in your project with
the index priority and fallthrough logic implemented by `uv`.  To make your
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
repositories (in fact, they link directly to the PyTorch packages).  They are
just an alternate index view that reduces some package conflicts.


## Developing

[issues]: https://github.com/lenskit/lkpy/issues
[workflow]: https://github.com/lenskit/lkpy/wiki/DevWorkflow

To contribute to LensKit, clone or fork the repository, get to work, and submit
a pull request.  We welcome contributions from anyone; if you are looking for a
place to get started, see the [issue tracker][issues].

Our development workflow is documented in [the wiki][workflow]; the wiki also
contains other information on *developing* LensKit. User-facing documentation is
at <https://lkpy.lenskit.org>.

[conda-lock]: https://github.com/conda-incubator/conda-lock
[lkdev]: https://github.com/lenskit/lkdev

We use [`uv`](https://astral.sh/uv/) for developing LensKit and managing
development environments.  Our `pyproject.toml` file contains the Python
development dependencies; you also need a working Rust compiler (typically via
[`rustup`](https://rustup.rs/)).  Before setting up to work on LensKit, you
therefore need:

- Git
- `uv`
- `rustup` and a working Rust compiler (`rustup install stable`)
- A working C compiler compatible with Python
    - On Windows, this is either Visual Studio (with C++ development) or the
      Visual C++ Build Tools. See the [Rustup Windows install
      instructions][rsu-win] for details.
    - On Mac, install Xcode.
    - On Linux, see your system package manager instructions.

<details>
<summary>Windows</summary>

On Windows, you can install dependencies (except for the Visual C++ tools) with `winget`:

```console
> winget install Git.Git astral-sh.uv Rustlang.Rustup
> rustup install stable-msvc
```
</details>

<details>
<summary>Mac</summary>

On Mac, you can install the dependencies with Homebrew:

```console
$ brew install git uv rustup
```
</details>

[rsu-win]: https://ehuss.github.io/rustup/installation/windows.html

Once you have the dependencies installed, set up your LensKit development
environment:

```console
$ uv venv -p 3.12
$ uv sync
```

If you want all extras (may not work on Windows), do:

```console
$ uv sync --all-extras
```

You can then activate the virtual environment to have the tools available and
run tools like `pytest`:

```console
$ . ./.venv/bin/activate
```

## Testing Changes

You should always test your changes by running the LensKit test suite:

    pytest tests

If you want to use your changes in a LensKit experiment, you can locally install
your modified LensKit into your experiment's environment.  We recommend using
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
