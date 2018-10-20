"""
Test utilities for LKPY tests.
"""

import os
import os.path
import tempfile
from contextlib import contextmanager


import pandas as pd
import pytest

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


ml100k = _ML100K()


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


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as dir:
        yield str(dir)
