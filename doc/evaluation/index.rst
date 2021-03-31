Evaluating Recommender Output
=============================

LensKit's evaluation support is based on post-processing the output of recommenders
and predictors.  The `batch utilities`_ provide support for generating these outputs.

We generally recommend using Jupyter_ notebooks for evaluation.

.. _batch utilities: batch.html
.. _Jupyter: https://jupyter.org

.. toctree::
   :caption: Evaluation Topics

   predict-metrics
   topn-metrics

Saving and Loading Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~

In our own experiments, we typically store the output of recommendation runs in LensKit
experiments in CSV or Parquet files, along with whatever parameters are relevant from
the configuration.
