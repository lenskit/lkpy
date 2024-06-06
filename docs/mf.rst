Classic Matrix Factorization
============================

LKPY provides classical matrix factorization implementations.

.. contents::
   :local:

Common Support
--------------

.. module:: lenskit.algorithms.mf_common

The :py:mod:`mf_common` module contains common support code for matrix factorization
algorithms.  This class, :py:class:`MFPredictor`,
defines the parameters that are estimated during the :py:meth:`.Algorithm.fit`
process on common matrix factorization algorithms.

.. autoclass:: MFPredictor
   :show-inheritance:
   :members:

.. autodata:: M

Alternating Least Squares
-------------------------

.. module:: lenskit.algorithms.als

LensKit provides alternating least squares implementations of matrix factorization suitable
for explicit feedback data.  These implementations are parallelized with Numba, and perform
best with the MKL from Conda.

.. autoclass:: BiasedMF
    :show-inheritance:
    :members:

.. autoclass:: ImplicitMF
    :show-inheritance:
    :members:

SciKit SVD
----------

.. module:: lenskit.algorithms.svd

This code implements a traditional SVD using scikit-learn.  It requires ``scikit-learn`` to
be installed in order to function.

.. autoclass:: BiasedSVD
    :show-inheritance:
    :members:

FunkSVD
-------

.. _FunkSVD: http://sifter.org/~simon/journal/20061211.html

.. module:: lenskit.funksvd

FunkSVD_ is an SVD-like matrix factorization that uses stochastic gradient
descent, configured much like coordinate descent, to train the user-feature and
item-feature matrices.  We generally don't recommend using it in new
applications or experiments; the ALS-based algorithms are less sensitive to
hyperparameters, and the TensorFlow algorithms provide more optimized gradient
descent training of the same prediction model.

.. note::
    FunkSVD must be installed separately from the lenskit-funksvd_ package.

.. versionchanged:: 2024.1
    FunkSVD moved from ``lenskit.algorithms.funksvd`` to ``lenskit.funksvd`` and
    is provided by a separate PyPI package ``lenskit-funksvd``.

.. _lenskit-funksvd: https://pypi.org/project/lenskit-funksvd

.. autoclass:: FunkSVD
    :show-inheritance:
    :members:
