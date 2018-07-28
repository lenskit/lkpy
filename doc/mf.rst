Classic Matrix Factorization
============================

LKPY provides classical matrix factorization implementations.

FunkSVD
-------

.. _FunkSVD: http://sifter.org/~simon/journal/20061211.html

.. module:: lenskit.algorithms.funksvd

FunkSVD_ is an SVD-like matrix factorization that uses stochastic gradient descent,
configured much like coordinate descent, to train the user-feature and item-feature
matrices.

.. autoclass:: FunkSVD
   :members:
