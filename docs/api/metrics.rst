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

Prediction Accuracy
-------------------

.. autosummary::
    :toctree:
    :nosignatures:
    :caption: Prediction Accuracy

    ~lenskit.metrics.RMSE
    ~lenskit.metrics.MAE
