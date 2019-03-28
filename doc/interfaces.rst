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
    
.. note::
    We are rethinking the ergonomics of this interface, and it may change in LensKit 0.6. We expect
    keep compatibility in the :py:func:`lenskit.batch.recommend` API, though.

.. autoclass:: Recommender
    :members:

Candidate Selection
-------------------

Some recommenders use a *candidate selector* to identify possible items to recommend.
These are also treated as algorithms, mainly so that they can memorize users' prior
ratings to exclude them from recommendation.

.. autoclass:: CandidateSelector
    :members:


Rating Prediction
-----------------

.. autoclass:: Predictor
   :members:
