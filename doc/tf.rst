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

.. warning::
    These implementations are not yet battle-tested --- they are here
    primarily for demonstration purposes at this time.

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
