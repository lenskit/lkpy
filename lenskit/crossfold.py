"""
Data set cross-folding.
"""

from collections import namedtuple
import logging

import numpy as np

TTPair = namedtuple('TTPair', ['train', 'test'])

_logger = logging.getLogger(__package__)

def partition_rows(data, partitions):
    """
    Partition a frame of ratings or other datainto train-test partitions.  This function does not
    care what kind of data is in `data`, so long as it is a Pandas DataFrame (or equivalent).

    :param data: a data frame containing ratings or other data you wish to partition.
    :type data: `pd.DataFrame` or equivalent
    :param partitions: the number of partitions to produce
    :type partitions: integer
    :rtype: iterator
    :returns: an iterator of train-test pairs
    """
    _logger.info('partitioning %d ratings into %d partitions', len(data), partitions)

    # create an array of indexes
    rows = np.arange(len(data))
    # shuffle the indices & split into partitions
    np.random.shuffle(rows)
    test_sets = np.array_split(rows, partitions)

    # convert each partition into a split
    for i, ts in enumerate(test_sets):
        test = data.iloc[ts,:]
        trains = test_sets[:i] + test_sets[(i+1):]
        train_idx = np.concatenate(trains)
        train = data.iloc[train_idx,:]
        yield TTPair(train, test)

def sample_rows(data, partitions, size, disjoint=True):
    """
    Sample train-test a frame of ratings into train-test partitions.  This function does not care what kind
    of data is in `data`, so long as it is a Pandas DataFrame (or equivalent).

    :param data: a data frame containing ratings or other data you wish to partition.
    :type data: `pd.DataFrame` or equivalent
    :param partitions: the number of partitions to produce
    :type partitions: integer
    :rtype: iterator
    :returns: an iterator of train-test pairs
    """

    if disjoint and partitions * size >= len(data):
        _logger.warn('wanted %d disjoint splits of %d each, but only have %d rows; partitioning',
                     partitions, size, len(data))
        for p in partition_rows(data, partitions):
            yield p
        return

    # create an array of indexes
    rows = np.arange(len(data))

    if disjoint:
        _logger.info('creating %d disjoint samples of size %d', partitions, size)
        # shuffle the indices & split into partitions
        np.random.shuffle(rows)

        # convert each partition into a split
        for i in range(partitions):
            start = i * size
            test = rows[start:start+size]
            train = np.concatenate((rows[:start], rows[start+size:]))
            yield TTPair(data.iloc[train,:], data.iloc[test,:])

    else:
        _logger.info('taking %d samples of size %d', partitions, size)
        for i in range(partitions):
            test = np.random.choice(rows, size, False)
            train = np.setdiff1d(rows, test, assume_unique=True)
            yield TTPair(data.iloc[train,:], data.iloc[test,:])
