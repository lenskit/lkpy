Evaluating Recommender Output
=============================

LensKit's evaluation support is based on post-processing the output of recommenders
and predictors.  The `batch utilities`_ provide support for generating these outputs.

We generally recommend using Jupyter_ notebooks for evaluation.

.. _batch utilities: batch.html
.. _Jupyter: https://jupyter.org

Loading Outputs
~~~~~~~~~~~~~~~

We typically store the output of recommendation runs in LensKit experiments in CSV or
Parquet files.  The :py:class:`lenskit.batch.MultiEval` class arranges to run a set
of algorithms over a set of data sets, and store the results in a collection of Parquet
files in a specified output directory.

There are several files:

``runs.parquet``
  The _runs_, algorithm-dataset combinations.  This file contains the names & any associated
  properties of each algorithm and data set run, such as a feature count.

``recommendations.parquet``
  The recommendations, with columns ``RunId``, ``user``, ``rank``, ``item``, and ``rating``.

``predictions.parquet``
  The rating predictions, if the test data includes ratings.

For example, if you want to examine nDCG by neighborhood count for a set of runs on a single
data set, you can do::

    import pandas as pd
    from lenskit.metrics import topn as lm

    runs = pd.read_parquet('eval-dir/runs.parquet')
    recs = pd.read_parquet('eval-dir/recs.parquet')
    meta = runs.loc[:, ['RunId', 'max_neighbors']]

    # compute each user's nDCG    
    user_ndcg = recs.groupby(['RunId', 'user']).rating.apply(lm.ndcg)
    user_ndcg = user_ndcg.reset_index(name='nDCG')
    # combine with metadata for feature count
    user_ndcg = pd.merge(user_ndcg, meta)
    # group and aggregate
    nbr_ndcg = user_ndcg.groupby('max_neighbors').nDCG.mean()
    nbr_ndcg.plot()
