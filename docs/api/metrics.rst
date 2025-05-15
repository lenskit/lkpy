Metrics and Analysis
====================

.. py:module:: lenskit.metrics

Base Interfaces
---------------

.. autosummary::
    :toctree:
    :nosignatures:
    :caption: Basic Interfaces

    ~lenskit.metrics.Metric
    ~lenskit.metrics.ListMetric
    ~lenskit.metrics.GlobalMetric
    ~lenskit.metrics.MetricFunction
    ~lenskit.metrics.RankingMetricBase

Bulk Analysis
-------------

.. autosummary::
    :toctree:
    :nosignatures:
    :caption: Bulk Analysis

    ~lenskit.metrics.RunAnalysis
    ~lenskit.metrics.RunAnalysisResult

Basic Statistics
----------------

.. autosummary::
    :toctree:
    :nosignatures:
    :caption: Basic Statistics

    ~lenskit.metrics.ListLength
    ~lenskit.metrics.TestItemCount

.. _metrics-topn:

Top-N Accuracy
--------------

.. autosummary::
    :toctree:
    :nosignatures:
    :caption: Top-N Accuracy

    ~lenskit.metrics.NDCG
    ~lenskit.metrics.RBP
    ~lenskit.metrics.Precision
    ~lenskit.metrics.Recall
    ~lenskit.metrics.RecipRank

List and Item Properties
------------------------

.. autosummary::
    :toctree:
    :nosignatures:
    :caption: List and Item Properties

    ~lenskit.metrics.MeanPopRank

Item Distributions
------------------

.. autosummary::
    :toctree:
    :nosignatures:
    :caption: Item Distributions

    ~lenskit.metrics.ExposureGini
    ~lenskit.metrics.ListGini

.. _metrics-predict:

Prediction Accuracy
-------------------

.. autosummary::
    :toctree:
    :nosignatures:
    :caption: Prediction Accuracy

    ~lenskit.metrics.RMSE
    ~lenskit.metrics.MAE

Rank Weights
------------

The rank weighting classes (:class:`RankWeight` and descendants) provide
flexible rank weights for use in evaluation metrics.  The rank-weighted top-*N*
metrics (:ref:`metrics-topn`) use these for weighting the recommendations.

.. autosummary::
    :toctree:
    :nosignatures:
    :caption: Rank Weights

    ~lenskit.metrics.RankWeight
    ~lenskit.metrics.GeometricRankWeight
    ~lenskit.metrics.LogRankWeight
