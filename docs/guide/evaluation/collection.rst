.. _eval-collection:

Collecting and Aggregating Metrics
==================================

.. py:currentmodule:: lenskit.metrics

Computing metrics over individual lists isn't enough for evaluating a
recommender system — we usually want to compute metrics for pipelines over
entire test sets.

For simple metrics, it is possible to do this yourself: call an appropriate
metric with each recommendation list and its corresponding truth, and compute
aggregate statistics over those metrics.  However, this has a few limitations:

- You need to implement correct logic to handle cases such as missing
  recommendations, missing truth data, etc.
- You need to implement the aggregation logic.
- Aggregation logic beyond simple statistical aggregates, such as computing the
  total number of unique recommended items across all recommendation lists,
  requires your code to have knowledge of the specific requirements and
  structure of each metric.

LensKit provides the :class:`MeasurementCollector` to help with all of this, and
to provide a unified way to collect metric measurements across all lists in a
recommendation run.

.. versionchanged:: 2026.1

    :class:`MeasurementCollector` was introduced in LensKit :ref:`2025.5.0`, and
    replaced :class:`RunAnalysis` as the primary metric analysis tool in LensKit
    :ref:`2026.1.0`.

Basic Principles
~~~~~~~~~~~~~~~~

A single measurement collector collects metrics for recommendation lists in a
**single run**: evaluating one pipeline on one test set.  The basic use pattern
is as follows:

1.  Create a :class:`MeasurementCollector`.
2.  Add metrics to collector with :meth:`~MeasurementCollector.add_metric`.
3.  Measure individual lists and their corresponding truth with
    :meth:`~MeasurementCollector.measure_list` or an entire collection of
    recommendations with :meth:`~MeasurementCollector.measure_collection`.
4.  Obtain individual list metrics with
    :meth:`~MeasurementCollector.list_metrics` (returning data frame with one
    row per list), or aggregate metrics and summary statistics with
    :meth:`~MeasurementCollector.summary_metrics`.

To measure multiple runs (e.g., the results of different recommendation
pipelines), there are three ways:

- Create a fresh :class:`MeasurementCollector` for each pipeline.
- Create an empty copy of a base collector with :meth:`~MeasurementCollector.empty_copy`.
- Reset the collector with :meth:`~MeasurementCollector.reset`.

The empty copy method is usually the easiest: create the measurement collector,
and then create a copy for each run to measure. :ref:`getting-started` provides
an example of using a measurement collector to measure two runs, and reporting
on the results.

.. note:: Design Goals for Aggregation

    Since there are many different ways of organizing experiments, and
    supporting complex aggregations inside the measurement collector would
    effectively be a reimplementation of the kinds of aggregation logic already
    supported by data frame libraries, we have focused LensKit's facilities on
    collecting and aggregating metrics within a single run, to produce summary
    statistics at the highest level that may be specific to recommendation.
    Further analysis can be done by collecting metric results (either summary or
    per-list) into larger data frames and analyzing with your preferred
    analytics library.

Example
~~~~~~~

If you have a dictionary of recommendation results in ``run_recs``, you can
measure them with:

.. code:: python

    base_mc = MeasurementCollector()
    base_mc.add_metric(NDCG(n=10))
    base_mc.add_metric(RBP(n=10))
    base_mc.add_metric(RecipRank(n=10))

    run_list_metrics = {}
    run_summaries = {}
    for name, recs in run_recs.items():
        mc = base_mc.empty_copy()
        mc.measure_collection(recs, test)
        run_list_metrics[name] = mc.list_metrics()
        run_summaries[name] = mc.summary_metrics()

    list_metrics = pd.concat(run_list_metrics, name=['recommender'])
    metrics = pd.DataFrame.from_dict(run_summaries, orient="index")
