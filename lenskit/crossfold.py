"""
Data set cross-folding.
"""

from collections import namedtuple
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from . import util

TTPair = namedtuple('TTPair', ['train', 'test'])
TTPair.__doc__ = 'Train-test pair (named tuple).'
TTPair.train.__doc__ = 'Train data for this pair.'
TTPair.test.__doc__ = 'Test data for this pair.'

_logger = logging.getLogger(__name__)


def partition_rows(data, partitions, *, rng_spec=None):
    """
    Partition a frame of ratings or other datainto train-test partitions.  This function does not
    care what kind of data is in `data`, so long as it is a Pandas DataFrame (or equivalent).

    Args:
        data(pandas.DataFrame):
            Ratings or other data you wish to partition.
        partitions(int):
            The number of partitions to produce.
        rng_spec:
            The random number generator or seed (see :py:func:`lenskit.util.rng`).

    Returns:
        iterator: an iterator of train-test pairs
    """
    _logger.info('partitioning %d ratings into %d partitions', len(data), partitions)

    # create an array of indexes
    rows = np.arange(len(data))
    # shuffle the indices & split into partitions
    rng = util.rng(rng_spec)
    rng.shuffle(rows)
    test_sets = np.array_split(rows, partitions)

    # convert each partition into a split
    for i, ts in enumerate(test_sets):
        test = data.iloc[ts, :]
        trains = test_sets[:i] + test_sets[(i + 1):]
        train_idx = np.concatenate(trains)
        train = data.iloc[train_idx, :]
        yield TTPair(train, test)


def sample_rows(data, partitions, size, disjoint=True, *, rng_spec=None):
    """
    Sample train-test a frame of ratings into train-test partitions.  This function does not care
    what kind of data is in `data`, so long as it is a Pandas DataFrame (or equivalent).

    We can loop over a sequence of train-test pairs::

        >>> from lenskit import datasets
        >>> ratings = datasets.MovieLens('data/ml-latest-small').ratings
        >>> for train, test in sample_rows(ratings, 5, 1000):
        ...     print(len(test))
        1000
        1000
        1000
        1000
        1000

    Sometimes for testing, it is useful to just get a single pair::

        >>> train, test = sample_rows(ratings, None, 1000)
        >>> len(test)
        1000
        >>> len(test) + len(train) - len(ratings)
        0

    Args:
        data(pandas.DataFrame):
            Data frame containing ratings or other data to partition.
        partitions(int or None):
            The number of partitions to produce.  If ``None``, produce a _single_ train-test
            pair instead of an iterator or list.
        size(int):
            The size of each sample.
        disjoint(bool):
            If ``True``, force samples to be disjoint.
        rng_spec:
            The random number generator or seed (see :py:func:`lenskit.util.rng`).

    Returns:
        iterator: An iterator of train-test pairs.
    """

    if partitions is None:
        test = data.sample(n=size)
        tr_mask = pd.Series(True, index=data.index)
        tr_mask.loc[test.index] = False
        train = data[tr_mask]
        return TTPair(train, test)

    if disjoint and partitions * size >= len(data):
        _logger.warning('wanted %d disjoint splits of %d each, but only have %d rows; partitioning',
                        partitions, size, len(data))
        return partition_rows(data, partitions)

    # create an array of indexes
    rows = np.arange(len(data))

    rng = util.rng(rng_spec)

    if disjoint:
        _logger.info('creating %d disjoint samples of size %d', partitions, size)
        ips = _disjoint_sample(rows, partitions, size, rng)

    else:
        _logger.info('taking %d samples of size %d', partitions, size)
        ips = _n_samples(rows, partitions, size, rng)

    return (TTPair(data.iloc[ip.train, :], data.iloc[ip.test, :]) for ip in ips)


def _disjoint_sample(xs, n, size, rng):
    # shuffle the indices & split into partitions
    rng.shuffle(xs)

    # convert each partition into a split
    for i in range(n):
        start = i * size
        test = xs[start:start + size]
        train = np.concatenate((xs[:start], xs[start + size:]))
        yield TTPair(train, test)


def _n_samples(xs, n, size, rng):
    for i in range(n):
        test = rng.choice(xs, size, False)
        train = np.setdiff1d(xs, test, assume_unique=True)
        yield TTPair(train, test)


class PartitionMethod(ABC):
    """
    Partition methods select test rows for a user or item.  Partition methods
    are callable; when called with a data frame, they return the test rows.
    """

    @abstractmethod
    def __call__(self, udf):
        """
        Subset a data frame.

        Args:
            udf(pandas.DataFrame):
                The input data frame of rows for a user or item.

        Returns:
            pandas.DataFrame:
                The data frame of test rows, a subset of ``udf``.
        """
        pass


class SampleN(PartitionMethod):
    """
    Randomly select a fixed number of test rows per user/item.

    Args:
        n(int): the number of test items to select
        rng: the random number generator or seed
    """

    def __init__(self, n, rng_spec=None):
        self.n = n
        self.rng = util.rng(rng_spec, legacy=True)

    def __call__(self, udf):
        return udf.sample(n=self.n, random_state=self.rng)


class SampleFrac(PartitionMethod):
    """
    Randomly select a fraction of test rows per user/item.

    Args:
        frac(float): the fraction items to select for testing.
    """
    def __init__(self, frac, rng_spec=None):
        self.fraction = frac
        self.rng = util.rng(rng_spec, legacy=True)

    def __call__(self, udf):
        return udf.sample(frac=self.fraction, random_state=self.rng)


class LastN(PartitionMethod):
    """
    Select a fixed number of test rows per user/item, based on ordering by a
    column.

    Args:
        n(int): The number of test items to select.
    """

    def __init__(self, n, col='timestamp'):
        self.n = n
        self.column = col

    def __call__(self, udf):
        return udf.sort_values(self.column).iloc[-self.n:]


class LastFrac(PartitionMethod):
    """
    Select a fraction of test rows per user/item.

    Args:
        frac(double): the fraction of items to select for testing.
    """
    def __init__(self, frac, col='timestamp'):
        self.fraction = frac
        self.column = col

    def __call__(self, udf):
        n = round(len(udf) * self.fraction)
        return udf.sort_values(self.column).iloc[-n:]


def partition_users(data, partitions: int, method: PartitionMethod, *, rng_spec=None):
    """
    Partition a frame of ratings or other data into train-test partitions user-by-user.
    This function does not care what kind of data is in `data`, so long as it is a Pandas DataFrame
    (or equivalent) and has a `user` column.

    Args:
        data(pandas.DataFrame): a data frame containing ratings or other data you wish to partition.
        partitions(int): the number of partitions to produce
        method(PartitionMethod): The method for selecting test rows for each user.
        rng_spec: The random number generator or seed (see :py:func:`lenskit.util.rng`).

    Returns
        iterator: an iterator of train-test pairs
    """

    user_col = data['user']
    users = user_col.unique()
    _logger.info('partitioning %d rows for %d users into %d partitions',
                 len(data), len(users), partitions)

    # create an array of indexes into user row
    rows = np.arange(len(users))
    # shuffle the indices & split into partitions
    rng = util.rng(rng_spec, legacy=True)
    rng.shuffle(rows)
    test_sets = np.array_split(rows, partitions)

    # convert each partition into a split
    for i, ts in enumerate(test_sets):
        # get our users!
        test_us = users[ts]
        # sample the data frame
        _logger.info('fold %d: selecting test ratings', i)
        ugf = data[data.user.isin(test_us)].groupby('user')
        test = ugf.apply(method)
        # get rid of the group index
        test = test.reset_index(0, drop=True)
        # now test is indexed on the data frame! so we can get the rest
        _logger.info('fold %d: partitioning training data', i)
        mask = pd.Series(True, index=data.index)
        mask[test.index] = False
        train = data[mask]
        yield TTPair(train, test)


def sample_users(data, partitions: int, size: int, method: PartitionMethod, disjoint=True, *,
                 rng_spec=None):
    """
    Create train-test partitions by sampling users.
    This function does not care what kind of data is in `data`, so long as it is
    a Pandas DataFrame (or equivalent) and has a `user` column.

    Args:
        data(pandas.DataFrame):
            Data frame containing ratings or other data you wish to partition.
        partitions(int):
            The number of partitions.
        size(int):
            The sample size.
        method(PartitionMethod):
            The method for obtaining user test ratings.
        rng_spec:
            The random number generator or seed (see :py:func:`lenskit.util.rng`).

    Returns:
        iterator: An iterator of train-test pairs (as :class:`TTPair` objects).
    """

    rng = util.rng(rng_spec, legacy=True)

    user_col = data['user']
    users = user_col.unique()
    if disjoint and partitions * size >= len(users):
        _logger.warning('cannot take %d disjoint samples of size %d from %d users',
                        partitions, size, len(users))
        yield from partition_users(data, partitions, method)
        return

    _logger.info('sampling %d users into %d partitions (n=%d)',
                 len(users), partitions, size)

    if disjoint:
        rng.shuffle(users)

    # generate our samples
    for i in range(partitions):
        # get our test users!
        if disjoint:
            test_us = users[i*size:(i+1)*size]
        else:
            test_us = rng.choice(users, size, False)

        # sample the data frame
        test = data[data.user.isin(test_us)].groupby('user').apply(method)
        # get rid of the group index
        test = test.reset_index(0, drop=True)
        # now test is indexed on the data frame! so we can get the rest
        rest = data.index.difference(test.index)
        train = data.loc[rest]
        yield TTPair(train, test)
