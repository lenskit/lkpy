Evaluating Top-N Rankings
=========================

.. py:currentmodule:: lenskit.metrics.ranking

.. _eval-topn:

The :py:mod:`lenskit.metrics.ranking` module contains the core top-*N* ranking
accuracy metrics (including rank-oblivious list metrics like precision, recall,
and hit rate).

Ranking metrics extend the :py:class:`RankingMetricBase` base class, often in
addition to :py:class:`ListMetric`, and return a score given a recommendation
list and a test rating list, both as :py:class:`item lists
<lenskit.data.ItemList>`; most metrics require the recommendation item list to
be :py:attr:`~lenskit.data.ItemList.ordered`.

All LensKit ranking metrics take ``n`` as a constructor argument to control the
list of the length that is considered; this allows multiple measurements (e.g.
HR@5 and HR@10) to be computed from a single set of rankings.

Metrics can be used on their own, but it is usually easiest to use them with
:class:`~lenskit.metrics.MeasurementCollector` to handle some of edge cases
around data availability, etc., as well as to support metric-specific
aggregation (see :ref:`eval-collection` for more details).

.. versionchanged:: 2026.1

    The argument for the list length has changed from ``k`` to ``n``, for
    consistency across LensKit.  ``k`` is kept as a deprecated alias until
    2027.1.

.. versionchanged:: 2025.1

    The top-N accuracy metric interface has changed to use item lists, and to
    be simpler to implement.

Descriptive Metrics
~~~~~~~~~~~~~~~~~~~

These metrics provide basic descriptive statistics of the recommendations and evaluation.

.. autoapisummary::
    :nosignatures:

    lenskit.metrics.ListLength
    lenskit.metrics.TestItemCount
    lenskit.metrics.UniqueItemCount

Included Effectiveness Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List and Set Metrics
--------------------

These metrics just look at the recommendation list and do not consider the rank
positions of items within it.

.. autoapisummary::
    :nosignatures:

    lenskit.metrics.Hit
    lenskit.metrics.Precision
    lenskit.metrics.Recall

Ranked List Metrics
-------------------

These metrics treat the recommendation list as a ranked list of items that may
or may not be relevant; some also support different item utilities (e.g. ratings
or graded relevance scores).

.. autoapisummary::
    :nosignatures:

    lenskit.metrics.RecipRank
    lenskit.metrics.RBP
    lenskit.metrics.IPSRBP
    lenskit.metrics.NDCG
    lenskit.metrics.DCG

Unbiased Evaluation
-------------------

Logged implicit feedback is missing-not-at-random: popular items are more likely
to be presented, and therefore more likely to be interacted with, so metrics
measured over such a log reward accuracy on popular items more than accuracy on
long-tail ones :cite:p:`yangUnbiasedOfflineRecommender2018`.
:class:`~lenskit.metrics.IPSRBP` corrects for this with inverse propensity
scoring, weighting each observed relevant item by the inverse of its propensity
to be observed.

Propensities are supplied by a :class:`~lenskit.metrics.PropensityModel`.
:class:`~lenskit.metrics.PopularityPropensity` estimates them from item
popularity in the training data, and
:func:`~lenskit.metrics.estimate_power_law_gamma` fits its exponent from a set
of recommendations::

    propensity = PopularityPropensity(split.train, gamma=1.87)
    mc.add_metric(RBP(n=10))
    mc.add_metric(IPSRBP(n=10, propensity=propensity))

Estimate propensities from the training data, not from the test data being
measured: the unbiasedness argument holds them fixed with respect to the
observations they weight.

.. versionadded:: 2026.3

Beyond Accuracy
~~~~~~~~~~~~~~~

These metrics measure **non-accuracy** properties of recommendation lists, such
as popularity/obscurity or diversity.


.. autoapisummary::
    :nosignatures:

    lenskit.metrics.MeanPopRank
    lenskit.metrics.ListGini
    lenskit.metrics.ExposureGini
