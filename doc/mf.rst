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

.. module:: lenskit.algorithms.funksvd

FunkSVD_ is an SVD-like matrix factorization that uses stochastic gradient
descent, configured much like coordinate descent, to train the user-feature and
item-feature matrices.  We generally don't recommend using it in new
applications or experiments; the ALS-based algorithms are less sensitive to
hyperparameters, and the TensorFlow algorithms provide more optimized gradient
descent training of the same prediction model.

.. autoclass:: FunkSVD
    :show-inheritance:
    :members:


References
----------

.. [HKV2008] Y. Hu, Y. Koren, and C. Volinsky. 2008.
    Collaborative Filtering for Implicit Feedback Datasets.
    In _Proceedings of the 2008 Eighth IEEE International Conference on Data Mining_, 263–272.
    DOI `10.1109/ICDM.2008.22 <http://dx.doi.org/10.1109/ICDM.2008.22>`_

.. [TPT2011] Gábor Takács, István Pilászy, and Domonkos Tikk. 2011. Applications of the
    Conjugate Gradient Method for Implicit Feedback Collaborative Filtering.

.. [ZWSP2008] Yunhong Zhou, Dennis Wilkinson, Robert Schreiber, and Rong Pan. 2008.
    Large-Scale Parallel Collaborative Filtering for the Netflix Prize.
    In +Algorithmic Aspects in Information and Management_, LNCS 5034, 337–348.
    DOI `10.1007/978-3-540-68880-8_32 <http://dx.doi.org/10.1007/978-3-540-68880-8_32>`_.
