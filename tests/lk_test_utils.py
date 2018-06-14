"""
Test utilities for LKPY tests.
"""

import os.path

import pandas as pd
import dask.dataframe as dd

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


ml_pandas = MLDataLoader(pd.read_csv)
ml_dask = MLDataLoader(dd.read_csv)
