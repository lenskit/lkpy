Implicit
========

.. module:: lenskit.implicit

This module provides a LensKit bridge to Ben Frederickson's :mod:`implicit` library
implementing some implicit-feedback recommender algorithms, with an emphasis
on matrix factorization.

It can be installed with the ``lenskit-implicit`` package::

    pip install lenskit-implicit
    conda install -c conda-forge lenskit-implicit\

.. note::
    This package is *not* necessary for working with implicit-feedback data,
    it is only for running the models from the :mod:`implicit`` library with LensKit.

.. autoclass:: ALS
    :members:

.. autoclass:: BPR
    :members:
