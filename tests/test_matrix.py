import scipy.sparse as sps
import scipy.linalg as sla
import numpy as np

import lenskit.matrix as lm

import lenskit.util.test as lktu

from pytest import approx


def test_sparse_matrix():
    ratings = lktu.ml_test.ratings
    mat, uidx, iidx = lm.sparse_ratings(ratings)

    assert mat.nrows == len(uidx)
    assert mat.nrows == ratings.user.nunique()
    assert mat.ncols == len(iidx)
    assert mat.ncols == ratings.item.nunique()

    # user indicators should correspond to user item counts
    ucounts = ratings.groupby('user').item.count()
    ucounts = ucounts.loc[uidx].cumsum()
    assert all(mat.rowptrs[1:] == ucounts.values)


def test_sparse_matrix_implicit():
    ratings = lktu.ml_test.ratings
    ratings = ratings.loc[:, ['user', 'item']]
    mat, uidx, iidx = lm.sparse_ratings(ratings)

    assert mat.nrows == len(uidx)
    assert mat.nrows == ratings.user.nunique()
    assert mat.ncols == len(iidx)
    assert mat.ncols == ratings.item.nunique()
    assert mat.values is None


def test_sparse_matrix_scipy():
    ratings = lktu.ml_test.ratings
    mat, uidx, iidx = lm.sparse_ratings(ratings, scipy=True)

    assert sps.issparse(mat)
    assert sps.isspmatrix_csr(mat)
    assert len(uidx) == ratings.user.nunique()
    assert len(iidx) == ratings.item.nunique()

    # user indicators should correspond to user item counts
    ucounts = ratings.groupby('user').item.count()
    ucounts = ucounts.loc[uidx].cumsum()
    assert all(mat.indptr[1:] == ucounts.values)


def test_sparse_matrix_scipy_implicit():
    ratings = lktu.ml_test.ratings
    ratings = ratings.loc[:, ['user', 'item']]
    mat, uidx, iidx = lm.sparse_ratings(ratings, scipy=True)

    assert sps.issparse(mat)
    assert sps.isspmatrix_csr(mat)
    assert len(uidx) == ratings.user.nunique()
    assert len(iidx) == ratings.item.nunique()

    assert all(mat.data == 1.0)
    
def test_ratings_surprise():
    ratings = lktu.ml_test.ratings
    ratings = ratings.loc[:, ['user', 'item']]
    dataset = lm.ratings_surprise(ratings)
    
    trainset = dataset.build_full_trainset()
    assert trainset.n_items == ratings.item.nunique()
    assert trainset.n_users == ratings.user.nunique()