Evaluating Rating Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:currentmodule:: lenskit.metrics.predict

.. _eval-predict-accuracy:

While rating prediction is no longer a widely-studied recommendation task,
LensKit provides support for evaluating predictions for completeness and
reproducing or comparing against historical research.

LensKit provides two :ref:`prediction accuracy metrics <metrics-predict>`:
:py:func:`RMSE` and :py:func:`MAE`.  They support both global (micro-averaged)
and per-user (macro-averaged) computation.

.. versionchanged:: 2025.1
    The prediction accuracy metric interface has changed to use item lists.


Calling Metrics
---------------

There are two ways to directly call a prediction accuracy metric:

* Pass two item lists, the first containing predicted ratings (as the list's
  :py:meth:`~lenskit.data.ItemLists.scores`) and the second containing
  ground-truth ratings as a ``rating`` field.

* Pass a single item list with scores and a ``rating`` field.

For evaluation, you will usually want to use :class:`~lenskit.metrics.RunAnalysis`,
which takes care of calling the prediction metric for you.

Missing Data
------------

There are two important missing data cases for evaluating predictions:

* Missing predictions (the test data has a rating for which the system could not
  generate a prediction).
* Missing ratings (the system generated a prediction with no corresponding test
  rating).

By default, LensKit throws an error in both of these cases, to help you catch
bad configurations.  We recommend using a fallback predictor, such setting up
:py:class:`~lenskit.basic.FallbackScorer` with
:py:class:`~lenskit.basic.BiasScorer`, when measuring rating predictions to
ensure that all items are scored.  The alternative design — ignoring missing
predictions — means that different scorers may be evaluated on different items,
and a scorer perform exceptionally well by only scoring items with high
confidence.

.. note::

    Both :func:`~lenskit.pipeline.topn_pipeline` and
    :func:`~lenskit.pipeline.predict_pipeline` default to using a
    :func:`~lenskit.basic.BiasScorer` as a fallback when rating prediction is
    enabled.

If you want to skip missing predictions, pass ``missing_scores="ignore"`` to the
metric function::

    RMSE(user_preds, user_ratings, missing_scores="ignore")

The corresponding ``missing_truth="ignore"`` will cause the metric to ignore
predictions with no corresponding rating (this case is unlikely to produce
unhelpful eval results, but may indicate a misconfiguration in how you determine
the items to score).
