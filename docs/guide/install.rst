.. _installation:

Installing LensKit
==================

We distribute LensKit through both PyPI and Conda (in the ``conda-forge`` channel).

Once you have installed LensKit, see `Getting Started`_.

.. _`Getting Started`: GettingStarted.html

Pip and uv
----------

.. _uv: https://docs.astral.sh/uv/

LensKit is distributed on PyPI in both source and binary formats.  You can use
use ``pip`` to install LensKit in a standard Python environment, such as a
virtual environment::

    pip install lenskit

We recommend using `uv`_, though::

    uv pip install lenskit

To add LensKit as a dependency to your project's ``pyproject.toml`` and virtual
environment::

    uv add lenskit

Conda and Pixi
--------------

You can also install LensKit from ``conda-forge``, using Pixi_::

    pixi add lenskit

Or ``conda``::

    conda install -c conda-forge lenskit

.. _Pixi: https://pixi.sh

Development Versions
--------------------

Our CI pipeline regularly builds development versions of LensKit as both wheels
and Conda packages.  These can be installed from:

- https://pypi.lenskit.org/lenskit-dev/ (PyPI index)
- https://prefix.dev/lenskit-dev/ (Conda channel)
