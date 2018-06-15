Crossfold preparation
=====================

The LKPY `crossfold` module provides support for preparing data sets for
cross-validation.  Crossfold methods are implemented as functions that operate
on data frames and return generators of `(train, test)` pairs
(:py:class:`lenskit.crossfold.TTPair` objects).  The train and test objects
in each pair are also data frames, suitable for evaluation or writing out to
a file.

Crossfold methods make minimal assumptions about their input data frames, so the
frames can be ratings, purchases, or whatever.  They do assume that each row
represents a single data point for the purpose of splitting and sampling.

.. autoclass:: lenskit.crossfold.TTPair

Row-based splitting
-------------------

The simplest preparation methods sample or partition the rows in the input frame.
A 5-fold :py:func:`lenskit.crossfold.partition_rows` split will result in 5
splits, each of which extracts 20% of the rows for testing and leaves 80% for
training.

.. autofunction:: lenskit.crossfold.partition_rows

.. autofunction:: lenskit.crossfold.sample_rows

User-based splitting
--------------------

It's often desirable to use users, instead of raw rows, as the basis for splitting
data.  This allows you to control the experimental conditions on a user-by-user basis,
e.g. by making sure each user is tested with the same number of ratings.  These methods
require that the input data frame have a `user` column with the user names or identifiers.

The algorithm used by each is as follows:

1. Sample or partition the set of user IDs into *n* sets of test users.
2. For each set of test users, select a set of that user's rows to be test rows.
3. Create a training set for each test set consisting of the non-selected rows from each
    of that set's test users, along with all rows from each non-test user.

.. autofunction:: lenskit.crossfold.partition_users

.. autofunction:: lenskit.crossfold.sample_users

Selecting user test rows
~~~~~~~~~~~~~~~~~~~~~~~~

These functions each take a `method` to decide how select each user's test rows. The method
is a function that takes a data frame (containing just the user's rows) and returns the
test rows.  This function is expected to preserve the index of the input data frame (which
happens by default with common means of implementing samples).

We provide several partition method factories:

.. autofunction:: lenskit.crossfold.SampleN
.. autofunction:: lenskit.crossfold.SampleFrac
