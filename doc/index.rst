LensKit
=======

LensKit is a set of Python tools for experimenting with and studying recommender
systems.  It provides support for training, running, and evaluating recommender
algorithms in a flexible fashion suitable for research and education.

LensKit for Python (also known as LKPY) is the successor to the Java-based
LensKit toolkit and a part of the LensKit project.

If you use Lenskit in published research, cite [LKPY]_.

.. [LKPY]
    Michael D. Ekstrand. 2018. The LKPY Package for Recommender Systems
    Experiments: Next-Generation Tools and Lessons Learned from the LensKit
    Project. *Computer Science Faculty Publications and Presentations*
    147. Boise State University.
    DOI:`10.18122/cs_facpubs/147/boisestate <https://dx.doi.org/10.18122/cs_facpubs/147/boisestate>`_.
    arXiv:`1809.03125 <https://arxiv.org/abs/1809.03125>`_ [cs.IR].

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
   performance
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

This material is based upon work supported by the National Science Foundation
under Grant No. IIS 17-51278. Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the author(s) and do not
necessarily reflect the views of the National Science Foundation.
