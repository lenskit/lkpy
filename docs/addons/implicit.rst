Implicit
========

.. module:: lenskit_implicit

This module provides a LensKit bridge to Ben Frederickson's implicit_ library
implementing some implicit-feedback recommender algorithms, with an emphasis
on matrix factorization.

It can be installed with the ``lenskit-implicit`` package::

    pip install lenskit-implicit
    conda install -c conda-forge lenskit-implicit

.. _implicit: https://implicit.readthedocs.io/en/latest/

.. autoclass:: ALS
    :members:

.. autoclass:: BPR
    :members:
