Algorithm Interfaces
====================

.. module:: lenskit.algorithms

LKPY's batch routines and utility support for managing algorithms expect algorithms
to implement consistent interfaces.  This page describes those interfaces.

The interfaces are realized as abstract base classes with the Python :py:mod:`abc` module.
Implementations must be registered with their interfaces, either by subclassing the interface
or by calling :py:meth:`abc.ABCMeta.register`.

Base Algorithm
--------------

Algorithms follow the SciKit fit-predict paradigm for estimators, except they know natively
how to work with Pandas objects.

The :py:class:`Algorithm` interface defines common methods.

.. autoclass:: Algorithm
    :members:

Recommendation
--------------

The :py:class:`Recommender` interface provides an interface to generating recommendations.  Not
all algorithms implement it; call :py:meth:`Recommender.adapt` on an algorithm to get a recommender
for any algorithm that at least implements :py:class:`Predictor`.  For example::

    pred = Bias(damping=5)
    rec = Recommender.adapt(pred)

.. autoclass:: Recommender
    :members:

Rating Prediction
-----------------

.. autoclass:: Predictor
   :members:
