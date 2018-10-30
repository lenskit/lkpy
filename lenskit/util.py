"""
Miscellaneous utility functions.
"""

import time
import pathlib

from numba import jitclass, njit, int32, double
import numpy as np


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
        else:
            return "{: 0.2f}s".format(elapsed)


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
