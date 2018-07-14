import tempfile
import logging

import pandas as pd
import numpy as np
import scipy as sp

import pytest
from pytest import mark

import lenskit.sharing as lks
import lk_test_utils as lktu

_log = logging.getLogger(__name__)


@pytest.fixture
def repo(request):
    with tempfile.TemporaryDirectory() as dir:
        yield lks.FileRepo(dir)


def test_share_data_frame(repo):
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': np.arange(3, 1, -0.5)})
    key = repo.share(df)
    assert key is not None
    _log.info('saved to %s', key)

    dfs = repo.resolve(key)

    assert dfs is not df
    assert len(dfs) == len(df)
    assert all(dfs.index == df.index)
    assert all(dfs.a == df.a)
    assert all(dfs.b == df.b)


def test_share_realistic_frame(repo):
    df = lktu.ml_pandas.ratings
    key = repo.share(df)
    assert key is not None
    _log.info('saved to %s', key)

    dfs = repo.resolve(key)

    assert dfs is not df
    assert len(dfs) == len(df)
    assert all(dfs.index == df.index)
    assert all(dfs.movieId == df.movieId)
    assert all(dfs.userId == df.userId)
    assert all(dfs.timestamp == df.timestamp)
    assert all(dfs.rating == df.rating)


def test_share_indexed_frame(repo):
    df = lktu.ml_pandas.ratings.set_index('movieId')
    key = repo.share(df)
    assert key is not None
    _log.info('saved to %s', key)

    dfs = repo.resolve(key)

    assert dfs is not df
    assert len(dfs) == len(df)
    assert all(dfs.index == df.index)
    assert all(dfs.userId == df.userId)
    assert all(dfs.timestamp == df.timestamp)
    assert all(dfs.rating == df.rating)


def test_share_prep_series(repo):
    s = pd.Series(np.random.randn(50), index=np.random.randint(0, 10000, 50), name='foo')

    tbl, schema = repo._to_table(s)
    assert b'lkpy' in schema.metadata
    assert b'pandas' in schema.metadata
    assert 'foo' in schema.names


def test_share_series(repo):
    s = pd.Series(np.random.randn(50))
    key = repo.share(s)
    assert key is not None
    _log.info('saved to %s', key)

    s2 = repo.resolve(key)
    assert s2 is not s
    assert isinstance(s2, pd.Series)
    assert len(s2) == len(s)
    assert all(s2.index == s.index)
    assert all(s2 == s)


def test_share_indexed_series(repo):
    s = pd.Series(np.random.randn(50), index=np.random.randint(0, 10000, 50))
    key = repo.share(s)
    assert key is not None
    _log.info('saved to %s', key)

    s2 = repo.resolve(key)
    assert s2 is not s
    assert isinstance(s2, pd.Series)
    assert len(s2) == len(s)
    assert all(s2.index == s.index)
    assert all(s2 == s)


def test_share_multi_series(repo):
    s = lktu.ml_pandas.renamed.ratings.set_index(['user', 'item']).rating
    key = repo.share(s)
    assert key is not None
    _log.info('saved to %s', key)

    s2 = repo.resolve(key)
    assert s2 is not s
    assert isinstance(s2, pd.Series)
    assert s2.name == 'rating'
    assert len(s2) == len(s)
    assert all(s2.index == s.index)
    assert all(s2 == s)


@mark.xfail
def test_share_array(repo):
    v = np.random.randn(50)
    key = repo.share(v)
    assert key is not None
    _log.info('saved to %s', key)

    v2 = repo.resolve(key)
    assert v2 is not v
    assert len(v2) == len(v)
    assert all(v2 == v)


@mark.xfail
def test_share_matrix(repo):
    m = np.random.randn(25, 50)
    key = repo.share(m)
    assert key is not None
    _log.info('saved to %s', key)

    m2 = repo.resolve(key)
    assert m2 is not m
    assert m2.ndim == m.ndim
    assert m2.shape == m.shape
    assert all(m2 == m)


@mark.parametrize('layout', ['csr', 'csc', 'coo'])
@mark.xfail
def test_share_sparse_matrix(repo, layout):
    from lenskit import matrix
    rm = matrix.sparse_ratings(lktu.ml_pandas.renamed.ratings)
    sm = rm.matrix

    key = repo.share(sm)
    assert key is not None
    _log.info('saved to %s', key)

    sm2 = repo.resolve(key)
    assert sm2 is not sm
    assert sm2.dtype == sm.dtype
    assert sm2.ndim == sm.ndim
    assert sm2.shape == sm.shape
    assert sm2.nnz == sm.nnz

    if layout == 'coo':
        assert all(sm2.row == sm.row)
        assert all(sm2.col == sm.col)
        assert all(sm2.data == sm.data)
    else:
        assert all(sm2.indptr == sm.indptr)
        assert all(sm2.indices == sm.indices)
        assert all(sm2.data == sm.data)


@mark.xfail
def test_share_index(repo):
    from lenskit import matrix
    rm = matrix.sparse_ratings(lktu.ml_pandas.renamed.ratings)
    iidx = rm.items

    key = repo.share(iidx)
    assert key is not None
    _log.info('saved to %s', key)

    i2 = repo.resolve(key)
    assert i2 is not iidx
    assert len(i2) == iidx
    assert all(i2.values == iidx.values)
    assert i2 == iidx


@mark.xfail
def test_share_str_index(repo):
    items = lktu.ml_pandas.ratings.movieId.unique()
    items = pd.Series(items).astype('str')
    iidx = pd.Index(items)

    key = repo.share(iidx)
    assert key is not None
    _log.info('saved to %s', key)

    i2 = repo.resolve(key)
    assert i2 is not iidx
    assert len(i2) == iidx
    assert all(i2.values == iidx.values)
    assert i2 == iidx
