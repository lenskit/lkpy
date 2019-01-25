Top-*N* Evaluation
~~~~~~~~~~~~~~~~~~

LensKit's support for top-*N* evaluation is in two parts, because there are some
subtle complexities that make it more dfficult to get the right data in the right
place for computing metrics correctly.

Top-*N* Analysis
================

.. module:: lenskit.topn

The :py:mod:`lenskit.topn` module contains the utilities for carrying out top-*N*
analysis, in conjucntion with :py:func:`lenskit.batch.recommend` and its wrapper
in :py:class:`lenskit.batch.MultiEval`.

The entry point to this is :py:class:`RecListAnalysis`.  This class encapsulates
an analysis with one or more metrics, and can apply it to data frames of recommendations.
An analysis requires two data frames: the recommendation frame contains the recommendations
themselves, and the truth frame contains the ground truth data for the users.  The
analysis is flexible with regards to the columns that identify individual recommendation
lists; usually these will consist of a user ID, data set identifier, and algorithm
identifier(s), but the analysis is configurable and its defaults make minimal assumptions.
The recommendation frame does need an ``item`` column with the recommended item IDs,
and it should be in order within a single recommendation list.

The truth frame should contain (a subset of) the columns identifying recommendation
lists, along with ``item`` and, if available, ``rating`` (if no rating is provided,
the metrics that need a rating value will assume a rating of 1 for every item present).
It can contain other items that custom metrics may find useful as well.

For example, a recommendation frame may contain:

* DataSet
* Partition
* Algorithm
* user
* item
* rank
* score

And the truth frame:

* DataSet
* user
* item
* rating

The analysis will use this truth as the relevant item data for measuring the accuracy of the
roecommendation lists.  Recommendations will be matched to test ratings by data set, user, 
and item, using :py:class:`RecListAnalysis` defaults.

.. autoclass:: RecListAnalysis
    :members:


Metrics
=======

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

The NDCG function estimates a utility score for a ranked list of recommendations.

.. autofunction:: ndcg


We also expose the internal DCG computation directly.

.. autofunction:: _dcg
