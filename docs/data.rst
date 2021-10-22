Data Utilities
--------------

.. module:: lenskit.data

These are general-purpose data processing utilities.

Building Ratings Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sparse_ratings
.. autoclass:: RatingMatrix

Sampling Utilities
~~~~~~~~~~~~~~~~~~

.. module:: lenskit.data.sampling

The :py:mod:`lenskit.data.sampling` module provides support functions for various
data sampling procedures for use in model training.


.. autofunction:: neg_sample
.. autofunction:: sample_unweighted
.. autofunction:: sample_weighted
