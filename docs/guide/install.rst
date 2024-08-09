Installing LensKit
---------------

To install the current release with Anaconda (recommended)::

    conda install -c conda-forge lenskit

You can also use ``pip`` to install LensKit in a stock Python environment,
such as a virtual environment::

    pip install lenskit

To use the latest development version, install directly from GitHub::

    pip install git+https://github.com/lenskit/lkpy

Then see `Getting Started`_.

.. _`Getting Started`: GettingStarted.html

.. note::
    If you install MKL-based BLAS in Conda, LensKit will use it to optimize
    several of its operations::

        conda install -c conda-forge libblas=*=*mkl
