.. LensKit documentation master file, created by
   sphinx-quickstart on Fri Jun 15 13:06:49 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LensKit
===================================

LensKit is a set of Python tools for experimenting with and studying recommender
systems.  It provides support for training, running, and evaluating recommender
algorithms in a flexible fashion suitable for research and education.

LensKit for Python (also known as LKPY) is the successor to the Java-based LensKit
project.

Installation
------------

To install the current release with Anaconda (recommended)::

    conda install -c lenskit lenskit

Or you can use ``pip``::

    pip install lenskit

To use the latest development version, install directly from GitHub::

    pip install git+https://github.com/lenskit/lkpy

Then see `Getting Started`_.

.. _`Getting Started`: quickstart.html

Resources
---------

- `Mailing list, etc. <https://lenskit.org/connect>`_
- `Source and issues on GitHub <https://github.com/lenskit/lkpy>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   GettingStarted
   crossfold
   batch
   evaluation/index
   algorithms
   util


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
