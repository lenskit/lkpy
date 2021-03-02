Install LensKit
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
    LensKit is optimized for MKL-based Anaconda installs. It works in other
    Python environments, but performance will usually suffer for some
    algorithms.  :py:class:`lenskit.algorithms.item_knn` is particularly
    affected by this.
