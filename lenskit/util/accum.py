"""
Accumulator support.
"""

import numpy as np
from numba import njit, jitclass, int32, double


@njit
def _swap(a, i, j):
    t = a[i]
    a[i] = a[j]
    a[j] = t


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
        _swap(keys, min, pos)
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
            _swap(keys, parent, current)
            current = parent
            parent = (current - 1) // 2


@njit
def _pair_downheap(pos: int, sp, limit, ks, vs):
    min = pos
    left = 2*pos + 1
    right = 2*pos + 2
    if left < limit and vs[sp + left] < vs[sp + min]:
        min = left
    if right < limit and vs[sp + right] < vs[sp + min]:
        min = right
    if min != pos:
        # we want to swap!
        _swap(vs, sp + pos, sp + min)
        _swap(ks, sp + pos, sp + min)
        _pair_downheap(min, sp, limit, ks, vs)


@njit
def _pair_upheap(pos, sp, ks, vs):
    parent = (pos - 1) // 2
    while pos > 0 and vs[sp + parent] > vs[sp + pos]:
        _swap(vs, sp + parent, sp + pos)
        _swap(ks, sp + parent, sp + pos)
        pos = parent
        parent = (pos - 1) // 2


@njit
def kvp_insert(sp, ep, limit, k, v, keys, vals):
    """
    Insert a value (with key) into a heap, only keeping the top values.

    Args:
        sp(int): the start of the heap
        ep(int): the current end of the heap
        limit(int): the maximum size of the heap
        k: the key
        v: the value (used for sorting)
        keys: the key array, must be at least sp+limit.
        vals: the value array, same size as keys

    Returns:
        int: the new ep
    """

    if ep - sp < limit:
        # insert into heap without size problems
        # put on end, then upheap
        keys[ep] = k
        vals[ep] = v
        _pair_upheap(ep - sp, sp, keys, vals)
        return ep + 1

    elif v > vals[sp]:
        # heap is full, but new value is larger than old min
        # stick it on the front, and downheap
        keys[sp] = k
        vals[sp] = v
        _pair_downheap(0, sp, limit, keys, vals)
        return ep

    else:
        # heap is full and new value doesn't belong
        return ep
