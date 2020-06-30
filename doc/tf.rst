TensorFlow Algorithms
=====================

.. module:: lenskit.algorithms.tf

.. _TensorFlow: https://tensorflow.org

LKPY provides several algorithm implementations, particularly matrix
factorization, using TensorFlow_.  These algorithms serve two purposes:

* Provide classic algorithms ready to use for recommendation or as
  baselines for new techniques.
* Demonstrate how to connect TensorFlow to LensKit for use in your own
  experiments.

.. toctree::
    :caption: Contents

Biased MF
---------

These models implement the standard biased matrix factorization model, like
:py:class:`lenskit.algorithms.als.BiasedMF`, but learn the model parameters
using TensorFlow's gradient descent instead of the alternating least squares
algorithm.

Bias-Based
~~~~~~~~~~

.. autoclass:: BiasedMF

Fully Integrated
~~~~~~~~~~~~~~~~

.. autoclass:: IntegratedBiasMF

Bayesian Personalized Rating
----------------------------

.. autoclass:: BPR
