Splitting Data
==============

.. module:: lenskit.splitting

The LKPY `splitting` module splits data sets for offline evaluation using
cross-validation and other strategies.  The various splitters are implemented as
functions that operate on a :class:`~lenskit.data.Dataset` and return one or
more train-test splits (as :class:`TTSplit` objects).

.. versionchanged:: 2024.1
    Data splitting was moved from ``lenskit.crossfold`` to the ``lenskit.splitting``
    module and functions were renamed and had their interfaces revised.

Experiment code should generally use these functions to prepare train-test files
for training and evaluating algorithms.  For example, the following will perform
a user-based 5-fold cross-validation as was the default in the old LensKit:

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
data. A 5-fold :func:`crossfold_records` split will result in 5 splits, each of
which extracts 20% of the user-item interaction records for testing and leaves
80% for training.

.. note::

    When a dataset has repeated interactions, these functions operate only on
    the *matrix* view of the data (user-item observations are deduplicated).
    Specifically, they operate on the results of calling
    :meth:`~lenskit.data.Dataset.interaction_matrix` with ``format="pandas"``
    and ``field="all"``.

.. autofunction:: crossfold_records

.. autofunction:: sample_records

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

.. autofunction:: crossfold_users

.. autofunction:: sample_users

Selecting user holdout rows
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions each take a `method` to decide how select each user's test rows. The method
is a function that takes an item list (containing just the user's rows) and returns the
test rows.

We provide several holdout method factories:

.. autofunction:: SampleN
.. autofunction:: SampleFrac
.. autofunction:: LastN
.. autofunction:: LastFrac

Utility Classes
---------------

.. autoclass:: lenskit.splitting.holdout.HoldoutMethod
   :members:
   :special-members: __call__

.. autoclass:: TTSplit
   :members:
