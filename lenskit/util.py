"""
Miscellaneous utility functions.
"""

import os
import os.path
import time
import pathlib
import warnings

from numba import jitclass, njit, int32, double
import numpy as np
import pandas as pd

try:
    import fastparquet
except ImportError:
    fastparquet = None


__os_fp = getattr(os, 'fspath', None)


@njit
def _ind_downheap(pos: int, size, keys, values):
    min = pos
    left = 2*pos + 1
    right = 2*pos + 2
    if left < size and values[keys[left]] < values[keys[min]]:
        min = left
    if right < size and values[keys[right]] < values[keys[min]]:
        min = right
    if min != pos:
        kt = keys[min]
        keys[min] = keys[pos]
        keys[pos] = kt
        _ind_downheap(min, size, keys, values)


@jitclass([
    ('nmax', int32),
    ('size', int32),
    ('keys', int32[:]),
    ('values', double[:])
])
class Accumulator:
    def __init__(self, values, nmax):
        self.values = values
        self.nmax = nmax
        self.size = 0
        self.keys = np.zeros(nmax + 1, dtype=np.int32)

    def __len__(self):
        return self.size

    def add(self, key):
        if key < 0 or key >= self.values.shape[0]:
            raise IndexError()
        self.keys[self.size] = key
        self._upheap(self.size)
        if self.size < self.nmax:
            self.size = self.size + 1
        else:
            # we are at capacity, we need to drop the smallest value
            self.keys[0] = self.keys[self.size]
            _ind_downheap(0, self.size, self.keys, self.values)

    def add_all(self, keys):
        for i in range(len(keys)):
            self.add(keys[i])

    def peek(self):
        if self.size > 0:
            return self.keys[0]
        else:
            return -1

    def remove(self):
        if self.size == 0:
            return -1

        top = self.keys[0]

        self.keys[0] = self.keys[self.size - 1]
        self.size = self.size - 1
        if self.size > 0:
            _ind_downheap(0, self.size, self.keys, self.values)
        return top

    def top_keys(self):
        keys = np.empty(self.size, dtype=np.int32)
        while self.size > 0:
            i = self.size - 1
            keys[i] = self.remove()
        return keys

    def _upheap(self, pos):
        keys = self.keys
        values = self.values
        current = pos
        parent = (current - 1) // 2
        while current > 0 and values[keys[parent]] > values[keys[current]]:
            # swap up
            kt = keys[parent]
            keys[parent] = keys[current]
            keys[current] = kt
            current = parent
            parent = (current - 1) // 2


class Stopwatch():
    start_time = None
    stop_time = None

    def __init__(self, start=True):
        if start:
            self.start()

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.stop_time = time.perf_counter()

    def elapsed(self):
        stop = self.stop_time
        if stop is None:
            stop = time.perf_counter()

        return stop - self.start_time

    def __str__(self):
        elapsed = self.elapsed()
        if elapsed < 1:
            return "{: 0.0f}ms".format(elapsed * 1000)
        elif elapsed > 60 * 60:
            h, m = divmod(elapsed, 60 * 60)
            m, s = divmod(m, 60)
            return "{:0.0f}h{:0.0f}m{:0.2f}s".format(h, m, s)
        elif elapsed > 60:
            m, s = divmod(elapsed, 60)
            return "{:0.0f}m{:0.2f}s".format(m, s)
        else:
            return "{:0.2f}s".format(elapsed)


def fspath(path):
    "Backport of :py:fun:`os.fspath` function for Python 3.5."
    if __os_fp:
        return __os_fp(path)
    else:
        return str(path)


def npz_path(path):
    path = pathlib.Path(path)
    p = path
    if not p.exists():
        p = path.with_name(path.name + '.npz')

    if not p.exists():
        p = path.with_suffix('.npz')

    if not p.exists():
        raise FileNotFoundError(path)

    return p


def read_df_detect(path):
    """
    Read a Pandas data frame, auto-detecting the file format based on filename suffix.
    The following file types are supported:

    CSV
        File has suffix ``.csv``, read with :py:fun:`pandas.read_csv`.
    Parquet
        File has suffix ``.parquet``, ``.parq``, or ``.pq``, read with
        :py:fun:`pandas.read_parquet`.
    """
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix in ('.parquet', '.parq', '.pq'):
        return pd.read_parquet(path)


def write_parquet(path, frame, append=False):
    """
    Write a Parquet file.

    Args:
        path(pathlib.Path): The path of the Parquet file to write.
        frame(pandas.DataFrame): The data to write.
        append(bool): Whether to append to the file or overwrite it.
    """
    fn = fspath(path)
    append = append and os.path.exists(fn)
    if fastparquet is not None:
        fastparquet.write(fn, frame, append=append, compression='snappy')
    elif append:
        warnings.warn('fastparquet not available, appending is slow')
        odf = pd.read_parquet(fn)
        pd.concat([odf, frame], ignore_index=True).to_parquet(fn)
    else:
        frame.to_parquet(fn)


class LastMemo:
    def __init__(self, func):
        self.function = func
        self.memory = None
        self.result = None

    def __call__(self, arg):
        if arg is not self.memory:
            self.result = self.function(arg)
            self.memory = arg

        return self.result
