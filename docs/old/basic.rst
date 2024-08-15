Basic and Utility Algorithms
============================

The :py:mod:`lenskit.algorithms.basic` module contains baseline and utility algorithms
for nonpersonalized recommendation and testing.


Most Popular Item Recommendation
--------------------------------

The :py:class:`PopScore` algorithm scores items by their popularity for enabling
most-popular-item recommendation.

.. module:: lenskit.algorithms.basic
.. autoclass:: PopScore
    :members:
    :show-inheritance:


Random Item Recommendation
--------------------------

The :py:class:`Random` algorithm implements random-item recommendation.

.. module:: lenskit.algorithms.basic
.. autoclass:: Random
    :members:
    :show-inheritance:


Unrated Item Candidate Selector
-------------------------------

:py:class:`UnratedItemCandidateSelector` is a candidate selector that remembers items
users have rated, and returns a candidate set consisting of all unrated items.  It is the
default candidate selector for :py:class:`TopN`.

.. module:: lenskit.algorithms.basic
.. autoclass:: UnratedItemCandidateSelector
    :members:
    :show-inheritance:


Fallback Predictor
------------------

The ``Fallback`` rating predictor is a simple hybrid that takes a list of composite algorithms,
and uses the first one to return a result to predict the rating for each item.

A common case is to fill in with :py:class:`Bias` when a primary predictor cannot score an item.

.. module:: lenskit.algorithms.basic
.. autoclass:: Fallback
   :members:
   :show-inheritance:

Memorized Predictor
-------------------

The ``Memorized`` recommender is primarily useful for test cases.  It memorizes a set of
rating predictions and returns them.

.. module:: lenskit.algorithms.basic
.. autoclass:: Memorized
   :members:
   :show-inheritance:
