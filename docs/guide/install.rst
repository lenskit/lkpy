Installing LensKit
==================

We distribute LensKit through both PyPI and Conda (in the ``conda-forge`` channel).

Once you have installed LensKit, see `Getting Started`_.

.. _`Getting Started`: GettingStarted.html


Conda and Pixi
--------------

To install the current release with Anaconda (recommended)::

    conda install -c conda-forge lenskit

If you are using Pixi_, you can add LensKit as a dependency to your project::

    pixi add lenskit

.. _Pixi: https://pixi.sh

.. note::

    To use the latest development LensKit in a Pixi project, add the following
    to your ``pixi.toml``:

    .. code-block:: toml

        [pypi-dependencies]
        lenskit = { git = "https://github.com/lenskit/lkpy.git" }

Pip
---

You can also use ``pip`` to install LensKit in a standard Python environment,
such as a virtual environment::

    pip install lenskit

To use the latest development version, install directly from GitHub::

    pip install git+https://github.com/lenskit/lkpy
