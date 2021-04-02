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

Identifying columns will be used to create two synthetic identifiers, `LKRecID` (the recommendation
list identifier) and `LKTruthID` (the truth list identifier), that are used in the internal data
frames.  Custom metric classes will see these on the data frames instead of other identifying columns.

.. autoclass:: RecListAnalysis
    :members:


Metrics
=======

.. module:: lenskit.metrics.topn

The :py:mod:`lenskit.metrics.topn` module contains metrics for evaluating top-*N*
recommendation lists.

Each of the top-*N* metrics supports an optional keyword argument ``k`` that specifies the expected list
length.  Recommendation lists are truncated to this length prior to measurment (so you can measure a
a metric at multiple values of ``k`` in a single analysis), and for recall-oriented metrics like
:py:func:`recall` and :py:func:`ndcg`, it normalizes the best-case possible items to ``k`` (because if
there are 10 relevant items, Recall@5 should be 1 when the list returns any 5 relevant items).
To use this, pass extra arguments to :py:meth:`RecListAnalysis.add_metric`::

    rla.add_metric(ndcg, k=5)
    rla.add_metric(ndcg, name='ndcg_10', k=10)

The default is to allow unbounded lists.  When using large recommendation lists, and users never have
more test ratings than there are recommended items, the default makes sense.


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
.. autofunction:: dcg

We also expose the internal DCG computation directly.

.. autofunction:: _dcg


Writing a Metric
----------------

A metric is a function that takes two positional parameters:


- ``recs``, a data frame of recommendations for a single recommendation list.
- ``truth``, a data frame of ground-truth data (usually ratings) for the user for whom the list was generated.

It can take additional keyword arguments that are passed through from :py:meth:`RecListAnalysis.add_metric`.
A metric then returns a single floating-point value; NaN is allowed.

Metrics can be further optimized with the *bulk interface*.  A bulk metric function takes ``recs``
and ``truth`` frames for the *entire set of recommendations*, with transformation (they have
``LKRecID`` and ``LKTruthID`` columns instead of other identifying columns), and returns a series 
whose index is ``LKRecID`` and values are the metric values for each list.  Further, the ``recs``
passed to a bulk implementation includes a 1-based *rank* for each recommendation.

The :py:func:`bulk_impl` function registers a bulk implementation of a metric::

    def metric(recs, truth):
        # normal metric implementation
        pass
    
    @bulk_impl(metric)
    def _bulk_metric(recs, truth):
        # bulk metric implementation

If a bulk implementation of a metric is available, and it is possible to use it, it will be used automatically
when the corresponding metric is passed to :py:meth:`RecListAnalysis.add_metric`.

.. autofunction: bulk_impl
