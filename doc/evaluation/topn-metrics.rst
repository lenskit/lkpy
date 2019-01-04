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

The DCG function estimates a utility score for a ranked list of recommendations.  The results
can be combined with ideal DCGs to compute nDCG.

.. autofunction:: dcg
.. autofunction:: compute_ideal_dcgs
