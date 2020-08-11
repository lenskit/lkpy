Basic and Utility Algorithms
============================

.. module:: lenskit.algorithms.basic

The :py:mod:`lenskit.algorithms.basic` and :py:mod:`lenskit.algorithms.bias` modules contain baseline and utility algorithms
for nonpersonalized recommendation and testing.

Personalized Mean Rating Prediction
-----------------------------------
.. module:: lenskit.algorithms.bias
.. autoclass:: Bias
    :members:
    :show-inheritance:


Most Popular Item Recommendation
--------------------------------

The :py:class:`Popular` algorithm implements most-popular-item recommendation.

.. autoclass:: Popular
    :members:
    :show-inheritance:


Random Item Recommendation
--------------------------

The :py:class:`Random` algorithm implements random-item recommendation.

.. autoclass:: Random
    :members:
    :show-inheritance:


Top-N Recommender
-----------------

The :py:class:`TopN` class implements a standard top-*N* recommender that wraps a
:py:class:`.Predictor` and :py:class:`.CandidateSelector` and returns the top *N*
candidate items by predicted rating.  It is the type of recommender returned by
:py:meth:`.Recommender.adapt` if the provided algorithm is not a recommender.

.. autoclass:: TopN
    :members:
    :show-inheritance:


Unrated Item Candidate Selector
-------------------------------

:py:class:`UnratedItemCandidateSelector` is a candidate selector that remembers items
users have rated, and returns a candidate set consisting of all unrated items.  It is the
default candidate selector for :py:class:`TopN`.

.. autoclass:: UnratedItemCandidateSelector
    :members:
    :show-inheritance:


Fallback Predictor
------------------

The ``Fallback`` rating predictor is a simple hybrid that takes a list of composite algorithms,
and uses the first one to return a result to predict the rating for each item.

A common case is to fill in with :py:class:`Bias` when a primary predictor cannot score an item.

.. autoclass:: Fallback
   :members:
   :show-inheritance:

Memorized Predictor
-------------------

The ``Memorized`` recommender is primarily useful for test cases.  It memorizes a set of
rating predictions and returns them.

.. autoclass:: Memorized
   :members:
   :show-inheritance:
