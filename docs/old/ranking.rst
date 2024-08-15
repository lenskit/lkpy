Ranking Methods
===============

.. module:: lenskit.algorithms.ranking

The :py:mod:`lenskit.algorithms.ranking` module contains various *ranking methods*:
algorithms that can use scores to produce ranks.  This includes primary rankers, like
:py:class:`TopN`, and some re-rankers as well.

Top-N Recommender
-----------------

The :py:class:`TopN` class implements a standard top-*N* recommender that wraps a
:py:class:`.Predictor` and :py:class:`.CandidateSelector` and returns the top *N*
candidate items by predicted rating.  It is the type of recommender returned by
:py:meth:`.Recommender.adapt` if the provided algorithm is not a recommender.

.. autoclass:: TopN
    :members:
    :show-inheritance:


Stochastic Recommenders
-----------------------

The :py:class:`PlackettLuce` class implements a stochastic recommender.  The underlying
relevance scores are kept the same, but the rankings are sampled from a Plackett-Luce
distribution instead using a deterministic top-N policy.

.. autoclass:: PlackettLuce
    :members:
    :show-inheritance:
