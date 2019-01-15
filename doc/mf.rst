Classic Matrix Factorization
============================

LKPY provides classical matrix factorization implementations.

.. contents::
   :local:

Common Support
--------------

.. module:: lenskit.algorithms.mf_common

The :py:mod:`mf_common` module contains common support code for matrix factorization
algorithms.

.. autoclass:: MFPredictor
   :members:

.. autoclass:: BiasMFPredictor
   :members:

Alternating Least Squares
-------------------------

.. module:: lenskit.algorithms.als

LensKit provides alternating least squares implementations of matrix factorization suitable
for explicit feedback data.  These implementations are parallelized with Numba, and perform
best with the MKL from Conda.

.. module:: lenskit.algorithms.als

.. autoclass:: BiasedMF
    :show-inheritance:
    :members:

.. autoclass:: ImplicitMF
    :show-inheritance:
    :members:

FunkSVD
-------

.. _FunkSVD: http://sifter.org/~simon/journal/20061211.html

.. module:: lenskit.algorithms.funksvd

FunkSVD_ is an SVD-like matrix factorization that uses stochastic gradient descent,
configured much like coordinate descent, to train the user-feature and item-feature
matrices.

.. autoclass:: FunkSVD
    :show-inheritance:
    :members:
