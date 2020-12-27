import scipy.sparse as sps
import scipy.linalg as sla
import numpy as np
import pandas as pd

import lenskit.matrix as lm

from lenskit.util.test import ml_test

from pytest import approx, mark


def test_sparse_matrix(rng):
    ratings = ml_test.ratings
    mat, uidx, iidx = lm.sparse_ratings(ratings)

    assert mat.nrows == len(uidx)
    assert mat.nrows == ratings.user.nunique()
    assert mat.ncols == len(iidx)
    assert mat.ncols == ratings.item.nunique()

    # user indicators should correspond to user item counts
    ucounts = ratings.groupby('user').item.count()
    ucounts = ucounts.loc[uidx].cumsum()
    assert all(mat.rowptrs[1:] == ucounts.values)

    # verify rating values
    ratings = ratings.set_index(['user', 'item'])
    for u in rng.choice(uidx, size=50):
        ui = uidx.get_loc(u)
        vs = mat.row_vs(ui)
        vs = pd.Series(vs, iidx[mat.row_cs(ui)])
        rates = ratings.loc[u]['rating']
        vs, rates = vs.align(rates)
        assert not any(vs.isna())
        assert not any(rates.isna())
        assert all(vs == rates)


def test_sparse_matrix_implicit():
    ratings = ml_test.ratings
    ratings = ratings.loc[:, ['user', 'item']]
    mat, uidx, iidx = lm.sparse_ratings(ratings)

    assert mat.nrows == len(uidx)
    assert mat.nrows == ratings.user.nunique()
    assert mat.ncols == len(iidx)
    assert mat.ncols == ratings.item.nunique()
    assert mat.values is None


@mark.parametrize(
    'format, sps_fmt_checker',
    [
        (True, sps.isspmatrix_csr),
        ('csr', sps.isspmatrix_csr),
        ('coo', sps.isspmatrix_coo),
    ],
)
def test_sparse_matrix_scipy(format, sps_fmt_checker):
    ratings = ml_test.ratings
    mat, uidx, iidx = lm.sparse_ratings(ratings, scipy=format)

    assert sps.issparse(mat)
    assert sps_fmt_checker(mat)
    assert len(uidx) == ratings.user.nunique()
    assert len(iidx) == ratings.item.nunique()

    # user indicators should correspond to user item counts
    ucounts = ratings.groupby('user').item.count()
    ucounts = ucounts.loc[uidx].cumsum()
    if sps.isspmatrix_coo(mat):
        mat = mat.tocsr()
    assert all(mat.indptr[1:] == ucounts.values)


def test_sparse_matrix_scipy_implicit():
    ratings = ml_test.ratings
    ratings = ratings.loc[:, ['user', 'item']]
    mat, uidx, iidx = lm.sparse_ratings(ratings, scipy=True)

    assert sps.issparse(mat)
    assert sps.isspmatrix_csr(mat)
    assert len(uidx) == ratings.user.nunique()
    assert len(iidx) == ratings.item.nunique()

    assert all(mat.data == 1.0)


def test_sparse_matrix_indexes(rng):
    ratings = ml_test.ratings
    uidx = pd.Index(rng.permutation(ratings['user'].unique()))
    iidx = pd.Index(rng.permutation(ratings['item'].unique()))

    mat, _uidx, _iidx = lm.sparse_ratings(ratings, users=uidx, items=iidx)

    assert _uidx is uidx
    assert _iidx is iidx
    assert len(_uidx) == ratings.user.nunique()
    assert len(_iidx) == ratings.item.nunique()

    # verify rating values
    ratings = ratings.set_index(['user', 'item'])
    for u in rng.choice(_uidx, size=50):
        ui = _uidx.get_loc(u)
        vs = mat.row_vs(ui)
        vs = pd.Series(vs, _iidx[mat.row_cs(ui)])
        rates = ratings.loc[u]['rating']
        vs, rates = vs.align(rates)
        assert not any(vs.isna())
        assert not any(rates.isna())
        assert all(vs == rates)
