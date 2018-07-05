"""
Test utilities for LKPY tests.
"""

import os
import os.path
from contextlib import contextmanager

import pandas as pd
import pytest

try:
    import dask.dataframe as dd

    have_dask = True
except ImportError as e:
    have_dask = False
    if os.getenv('CI'):
        raise e

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
if have_dask:
    ml_dask = MLDataLoader(dd.read_csv)


def dask_test(f):
    """
    Decorator for test cases that require Dask.
    :param f: The test function
    :return:
    """
    return pytest.mark.skipif(not have_dask, reason='dask is not available')(f)
