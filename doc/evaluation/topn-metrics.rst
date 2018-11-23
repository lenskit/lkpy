Top-*N* Accuracy Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: lenskit.metrics.topn

The :py:mod:`lenskit.metrics.topn` module contains metrics for evaluating top-*N*
recommendation lists.

Classification Metrics
----------------------

These metrics treat the recommendation list as a classification of relevant items.

.. autofunction:: precision
.. autofunction:: recall

Ranked List Metrics
-------------------

These metrics treat the recommendation list as a ranked list of items that may or may not
be relevant.

.. autofunction:: recip_rank

Utility Metrics
---------------

The nDCG function estimates a utility score for a ranked list of recommendations.

.. autofunction:: ndcg
