import scipy.sparse as sps

import lenskit.matrix as lm

import lk_test_utils as lktu


def test_sparse_matrix():
    ratings = lktu.ml_pandas.renamed.ratings
    mat, uidx, iidx = lm.sparse_ratings(ratings)

    assert sps.issparse(mat)
    assert sps.isspmatrix_csr(mat)
    assert len(uidx) == ratings.user.nunique()
    assert len(iidx) == ratings.item.nunique()

    # user indicators should correspond to user item counts
    ucounts = ratings.groupby('user').item.count()
    ucounts = ucounts.loc[uidx].cumsum()
    assert all(mat.indptr[1:] == ucounts.values)


def test_sparse_matrix_csc():
    ratings = lktu.ml_pandas.renamed.ratings
    mat, uidx, iidx = lm.sparse_ratings(ratings, layout='csc')

    assert sps.issparse(mat)
    assert sps.isspmatrix_csc(mat)
    assert len(uidx) == ratings.user.nunique()
    assert len(iidx) == ratings.item.nunique()

    # user indicators should correspond to user item counts
    icounts = ratings.groupby('item').user.count()
    icounts = icounts.loc[iidx].cumsum()
    assert all(mat.indptr[1:] == icounts.values)
