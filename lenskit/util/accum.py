"""
Accumulator support.
"""

from numba import njit
from .array import swap


@njit
def _pair_downheap(pos: int, sp, limit, ks, vs):
    finished = False
    while not finished:
        min = pos
        left = 2*pos + 1
        right = 2*pos + 2
        if left < limit and vs[sp + left] < vs[sp + min]:
            min = left
        if right < limit and vs[sp + right] < vs[sp + min]:
            min = right
        if min != pos:
            # we want to swap!
            swap(vs, sp + pos, sp + min)
            swap(ks, sp + pos, sp + min)
            pos = min
        else:
            finished = True


@njit
def _pair_upheap(pos, sp, ks, vs):
    parent = (pos - 1) // 2
    while pos > 0 and vs[sp + parent] > vs[sp + pos]:
        swap(vs, sp + parent, sp + pos)
        swap(ks, sp + parent, sp + pos)
        pos = parent
        parent = (pos - 1) // 2


@njit
def kvp_minheap_insert(sp, ep, limit, k, v, keys, vals):
    """
    Insert a value (with key) into a heap-organized array subset, only keeping the top values.

    Args:
        sp(int): the start of the heap (inclusive)
        ep(int): the current end of the heap (exclusive)
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


@njit
def kvp_minheap_sort(sp, ep, keys, vals):
    """
    Sort a heap-organized array subset by decreasing values.

    Args:
        sp(int): the start of the heap (inclusive).
        ep(int): the end of the heap (exclusive).
        keys: the key array
        vals: the value array
    """

    for i in range(ep-1, sp, -1):
        swap(keys, i, sp)
        swap(vals, i, sp)
        _pair_downheap(0, sp, i-sp, keys, vals)
