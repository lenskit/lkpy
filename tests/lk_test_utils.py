"""
Test utilities for LKPY tests.
"""

import os
import os.path
import tempfile
import pathlib
import logging
from contextlib import contextmanager

import pandas as pd
import pytest

_log = logging.getLogger('lktu')

ml_dir = os.path.join(os.path.dirname(__file__), '../ml-latest-small')


class Renamer:
    def __init__(self, dl):
        self._dl = dl

    def __getattribute__(self, name):
        dl = object.__getattribute__(self, '_dl')
        df = getattr(dl, name)
        return df.rename(columns={'userId': 'user', 'movieId': 'item'})


class MLDataLoader:
    _ratings = None
    _movies = None
    _tags = None

    def __init__(self, reader):
        self._read = reader

    @property
    def ratings(self):
        if self._ratings is None:
            self._ratings = self._read(os.path.join(ml_dir, 'ratings.csv'))
        return self._ratings

    @property
    def movies(self):
        if self._movies is None:
            self._movies = self._read(os.path.join(ml_dir, 'movies.csv'))
        return self._movies

    @property
    def tags(self):
        if self.tags is None:
            self.tags = self._read(os.path.join(ml_dir, 'tags.csv'))
        return self.tags

    @property
    def renamed(self):
        return Renamer(self)


class _ML100K:
    @property
    def location(self):
        return os.path.expanduser(os.environ.get('ML100K_DIR', 'ml-100k'))

    @property
    def rating_file(self):
        return os.path.join(self.location, 'u.data')

    @property
    def available(self):
        return os.path.exists(self.rating_file)

    def load_ratings(self):
        return pd.read_csv(self.rating_file, sep='\t',
                           names=['user', 'item', 'rating', 'timestamp'])


def ml_sample():
    ratings = ml_pandas.renamed.ratings
    icounts = ratings.groupby('item').rating.count()
    top = icounts.nlargest(500)
    ratings = ratings.set_index('item')
    top_rates = ratings.loc[top.index, :]
    _log.info('top 500 items yield %d of %d ratings', len(top_rates), len(ratings))
    return top_rates.reset_index()


ml100k = _ML100K()


wantjit = pytest.mark.skipif('NUMBA_DISABLE_JIT' in os.environ,
                             reason='JIT required')


@contextmanager
def envvars(**vars):
    save = {}
    for k in vars.keys():
        if k in os.environ:
            save[k] = os.environ[k]
        else:
            save[k] = None
        os.environ[k] = vars[k]
    try:
        yield
    finally:
        for k, v in save.items():
            if v is None:
                del os.environ[k]
            else:
                os.environ[k] = v


ml_pandas = MLDataLoader(pd.read_csv)


def norm_path(path):
    if isinstance(path, pathlib.Path):
        return path
    elif hasattr(path, '__fspath__'):
        return pathlib.Path(path.__fspath__())
    elif isinstance(path, str):
        return pathlib.Path(str)
    else:
        raise ValueError('invalid path: ' + repr(path))
