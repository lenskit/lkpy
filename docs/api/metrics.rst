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

Prediction Accuracy
-------------------

.. autosummary::
    :toctree:
    :nosignatures:
    :caption: Prediction Accuracy

    ~lenskit.metrics.RMSE
    ~lenskit.metrics.MAE
