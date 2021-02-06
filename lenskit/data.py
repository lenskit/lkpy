"""
Data manipulation routines.
"""

from collections import namedtuple
import logging

import scipy.sparse as sps
import numpy as np
import pandas as pd
from csr import CSR

_log = logging.getLogger(__name__)

RatingMatrix = namedtuple('RatingMatrix', ['matrix', 'users', 'items'])
RatingMatrix.__doc__ = """
A rating matrix with associated indices.

Attributes:
    matrix(CSR or scipy.sparse.csr_matrix):
        The rating matrix, with users on rows and items on columns.
    users(pandas.Index): mapping from user IDs to row numbers.
    items(pandas.Index): mapping from item IDs to column numbers.
"""


def sparse_ratings(ratings, scipy=False, *, users=None, items=None):
    """
    Convert a rating table to a sparse matrix of ratings.

    Args:
        ratings(pandas.DataFrame): a data table of (user, item, rating) triples.
        scipy(bool):
            if ``True`` or ``'csr'``, return a SciPy csr matrix instead of
            :py:class:`CSR`. if ``'coo'``, return a SciPy coo matrix.
        users(pandas.Index): an index of user IDs.
        items(pandas.Index): an index of items IDs.

    Returns:
        RatingMatrix:
            a named tuple containing the sparse matrix, user index, and item index.
    """
    if users is None:
        users = pd.Index(np.unique(ratings.user), name='user')

    if items is None:
        items = pd.Index(np.unique(ratings.item), name='item')

    _log.debug('creating matrix with %d ratings for %d items by %d users',
               len(ratings), len(items), len(users))

    row_ind = users.get_indexer(ratings.user).astype(np.intc)
    if np.any(row_ind < 0):
        raise ValueError('provided user index does not cover all users')
    col_ind = items.get_indexer(ratings.item).astype(np.intc)
    if np.any(col_ind < 0):
        raise ValueError('provided item index does not cover all users')

    if 'rating' in ratings.columns:
        vals = np.require(ratings.rating.values, np.float64)
    else:
        vals = None

    if scipy == 'coo':
        matrix = sps.coo_matrix(
            (vals, (row_ind, col_ind)), shape=(len(users), len(items))
        )
    else:
        matrix = CSR.from_coo(row_ind, col_ind, vals, (len(users), len(items)))
        if scipy:
            matrix = matrix.to_scipy()

    return RatingMatrix(matrix, users, items)
