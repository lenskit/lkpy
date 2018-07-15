import sys
import os.path
import tempfile
import logging
import uuid
import subprocess
from contextlib import closing

import pandas as pd
import numpy as np
import scipy as sp

import pytest
from pytest import mark

import lenskit.sharing as lks
import lk_test_utils as lktu

_log = logging.getLogger(__name__)


@pytest.fixture(params=['file', 'plasma'])
def repo(request):
    if request.param == 'file':
        with lks.FileRepo() as repo:
            _log.info('using directory %s', repo.dir)
            yield repo
    elif request.param == 'plasma':
        if sys.platform == 'win32':
            raise pytest.skip('Plasma unsupported on Win32')

        with tempfile.TemporaryDirectory() as dir:
            proc = None
            try:
                proc = subprocess.Popen(['plasma_store', '-m', str(256*1024*1024),
                                         '-s', os.path.join(dir, 'plasma.sock')])
                with lks.PlasmaRepo(os.path.join(dir, 'plasma.sock')) as repo:
                    yield repo
            finally:
                if proc is not None:
                    proc.terminate()


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


def test_share_array(repo):
    v = np.random.randn(50)
    key = repo.share(v)
    assert key is not None
    _log.info('saved to %s', key)

    v2 = repo.resolve(key)
    assert v2 is not v
    assert len(v2) == len(v)
    assert all(v2 == v)


def test_share_string_array(repo):
    v = np.array([str(uuid.uuid4()) for i in range(20)])
    key = repo.share(v)
    assert key is not None
    _log.info('saved to %s', key)

    v2 = repo.resolve(key)
    assert v2 is not v
    assert len(v2) == len(v)
    assert all(v2 == v)


def test_share_string_series(repo):
    v = pd.Series([str(uuid.uuid4()) for i in range(20)])
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
def test_share_sparse_matrix(repo, layout):
    from lenskit import matrix
    rm = matrix.sparse_ratings(lktu.ml_pandas.renamed.ratings, layout=layout)
    sm = rm.matrix

    sm_coo = sm.tocoo()

    key = repo.share(sm_coo)
    assert key is not None
    _log.info('saved to %s', key)

    sm2_coo = repo.resolve(key)
    assert sm2_coo is not sm_coo
    sm2 = getattr(sm2_coo, 'to' + layout)()
    assert sm2.dtype == sm.dtype
    assert sm2.ndim == sm.ndim
    assert sm2.shape == sm.shape
    assert sm2.nnz == sm.nnz

    if layout == 'coo':
        assert sp.sparse.isspmatrix_coo(sm)
        assert all(sm2.row == sm.row)
        assert all(sm2.col == sm.col)
        assert all(sm2.data == sm.data)
    else:
        assert all(sm2.indptr == sm.indptr)
        assert all(sm2.indices == sm.indices)
        assert all(sm2.data == sm.data)
