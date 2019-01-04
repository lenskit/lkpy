Basic and Utility Algorithms
============================

.. module:: lenskit.algorithms.basic

The :py:mod:`lenskit.algorithms.basic` module contains baseline and utility algorithms
for nonpersonalized recommendation and testing.

Personalized Mean Rating Prediction
-----------------------------------

.. autoclass:: Bias
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
