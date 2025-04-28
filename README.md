# Python recommendation tools

[![Automatic Tests](https://github.com/lenskit/lkpy/actions/workflows/test.yml/badge.svg)](https://github.com/lenskit/lkpy/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/lenskit/lkpy/graph/badge.svg?token=DaGn7NFM2P)](https://codecov.io/gh/lenskit/lkpy)
[![Scientific Python Ecosystem Coordination](https://img.shields.io/badge/SPEC-0,7-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/)
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

To install the current release with Anaconda (recommended):

    conda install -c conda-forge lenskit

If you use Pixi, you can add it to your project:

    pixi add lenskit

Or you can use `pip` (or `uv`):

    pip install lenskit

To use the latest development version, install directly from GitHub:

    pip install -U git+https://github.com/lenskit/lkpy

Then see [Getting Started](https://lkpy.lenskit.org/stable/guide/GettingStarted.html)

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
