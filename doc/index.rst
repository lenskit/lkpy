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

.. _`Getting Started`: GettingStarted.html

Resources
---------

- `Mailing list, etc. <https://lenskit.org/connect>`_
- `Source and issues on GitHub <https://github.com/lenskit/lkpy>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   GettingStarted
   interfaces
   crossfold
   batch
   evaluation/index
   datasets
   algorithms
   random
   util
   diagnostics
   impl-tips
   internals
   Release Notes <https://github.com/lenskit/lkpy/releases>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Acknowledgements
================

This material is based upon work supported by the National Science Foundation under Grant No. IIS 17-51278.
Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
