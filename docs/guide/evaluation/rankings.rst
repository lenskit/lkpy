Evaluating Top-N Rankings
=========================

.. py:currentmodule:: lenskit.metrics.ranking

.. _eval-topn:

The :py:mod:`lenskit.metrics.ranking` module contains the core top-*N* ranking
accuracy metrics (including rank-oblivious list metrics like precision, recall,
and hit rate).

Ranking metrics extend the :py:class:`RankingMetric` base class, and are
callable objects that take the recommendation list and the test ratings, both as
:py:class:`~lenskit.data.ItemList`, and return the score; most metrics require
the recommendation item list to be :py:attr:`~lenskit.data.ItemList.ordered`.

All LensKit ranking metrics take `k` as a constructor argument to control the
list of the length that is considered; this allows multiple measurements (e.g.
HR@5 and HR@10) to be computed from a single set of rankings.

.. versionchanged:: 2025.1
    The top-N accuracy metric interface has changed to use item lists, and to
    be simpler to implement.

Included Effectiveness Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List and Set Metrics
--------------------

These metrics just look at the recommendation list and do not consider the rank
positions of items within it.

.. autosummary::
    :nosignatures:

    ~lenskit.metrics.Hit
    ~lenskit.metrics.Precision
    ~lenskit.metrics.Recall

Ranked List Metrics
-------------------

These metrics treat the recommendation list as a ranked list of items that may
or may not be relevant; some also support different item utilities (e.g. ratings
or graded relevance scores).

.. autosummary::
    :nosignatures:

    ~lenskit.metrics.RecipRank
    ~lenskit.metrics.RBP
    ~lenskit.metrics.NDCG
