Splitting Data
==============

.. py:currentmodule:: lenskit.splitting

The :mod:`~lenskit.splitting` module splits data sets for offline evaluation
using cross-validation and other strategies.  The various splitters are
implemented as functions that operate on a :class:`~lenskit.data.Dataset` and
return one or more train-test splits (as :class:`.TTSplit` objects).

.. versionchanged:: 2024.1
    Data splitting was moved from ``lenskit.crossfold`` to the :mod:`lenskit.splitting`
    module and functions were renamed and had their interfaces revised.

Experiment code should generally use these functions to prepare train-test files
for training and evaluating algorithms.  For example, the following will perform
a user-based 5-fold cross-validation as was the default in older versions of
LensKit:

.. code:: python

    import pandas as pd
    from lenskit.data import load_movielens
    from lenskit.splitting import crossfold_users, SampleN, dict_to_df
    dataset = load_movielens('data/ml-20m.zip')
    for i, tp in enumerate(crossfold_users(ratings, 5, SampleN(5))):
        tp.train_df.to_parquet(f'ml-20m.exp/train-{i}.parquet')
        tp.test_df.to_parquet(f'ml-20m.exp/test-{i}.parquet')

Record-based Random Splitting
-----------------------------

The simplest preparation methods sample or partition the records in the input
data. A 5-fold :func:`.crossfold_records` split will result in 5 splits, each of
which extracts 20% of the user-item interaction records for testing and leaves
80% for training.  There are two record-based random splitting functions:

* :func:`.crossfold_records` partitions ratings or interactions into 5
  equal-sized splits.
* :func:`.sample_records` produces 1 or more disjoint samples of the ratings for
  testing.

.. note::

    When a dataset has repeated interactions, these functions operate only on
    the *matrix* view of the data (user-item observations are deduplicated).
    Specifically, they operate on the results of calling
    :meth:`~lenskit.data.Dataset.interaction_matrix` with ``format="pandas"``
    and ``field="all"``.

User-based Splitting
--------------------

It's often desirable to use users, instead of raw rows, as the basis for
splitting data.  This allows you to control the experimental conditions on a
user-by-user basis, e.g. by making sure each user is tested with the same number
of ratings.  These methods require that the input data frame have a `user`
column with the user names or identifiers.

The algorithm used by each is as follows:

1.  Sample or partition the set of user IDs into *n* sets of test users.
2.  For each set of test users, select a set of that user's rows to be test rows.
3.  Create a training set for each test set consisting of the non-selected rows
    from each of that set's test users, along with all rows from each non-test
    user.

As with record-based splitting, there are both cross-folding (partition all
users into disjoint sets) and sampling (compute one or more disjoint sets of
test users).

* :func:`.crossfold_users`
* :func:`.sample_users`

Selecting user holdout rows
~~~~~~~~~~~~~~~~~~~~~~~~~~~

User-based splitting requires a mechanism to split a test user's interactions
into the actual test data and the training or query data for that user.  The
user-based splitting functions therefore take a :class:`holdout method
<HoldoutMethod>` (the ``method`` parameter) to do that partitioning.  The method
is just a callable that takes an item list of the user's interactions and
returns the test interactions.

We provide several holdout implementations, implemented as classes that take
the holdout's configuration (e.g. the number of test ratings per user) and
return callable objects to do the holdout:

.. autosummary::
    SampleN
    SampleFrac
    LastN
    LastFrac
