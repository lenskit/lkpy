# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import assume, given, settings

from lenskit.util.kvp import KVPHeap


def test_kvp_add_to_empty():
    ks = np.empty(10, dtype=np.int32)
    vs = np.empty(10)

    # insert an item
    kvp = KVPHeap(0, 0, 10, ks, vs)
    n = kvp.insert(5, 3.0)

    # ep has moved
    assert n == 1
    assert kvp.ep == 1

    # item is there
    assert ks[0] == 5
    assert vs[0] == 3.0


def test_kvp_add_larger():
    ks = np.empty(10, dtype=np.int32)
    vs = np.empty(10)

    # insert an item
    kvp = KVPHeap(0, 0, 10, ks, vs)
    n = kvp.insert(5, 3.0)
    n = kvp.insert(1, 6.0)

    # ep has moved
    assert n == 2
    assert kvp.ep == 2

    # data is there
    assert all(ks[:2] == [5, 1])
    assert all(vs[:2] == [3.0, 6.0])


def test_kvp_add_smaller():
    ks = np.empty(10, dtype=np.int32)
    vs = np.empty(10)

    # insert an item
    kvp = KVPHeap(0, 0, 10, ks, vs)
    n = kvp.insert(5, 3.0)
    n = kvp.insert(1, 1.0)

    # ep has moved
    assert n == 2

    # data is there
    assert all(ks[:2] == [1, 5])
    assert all(vs[:2] == [1.0, 3.0])


@given(st.integers(10, 100), st.data())
def test_kvp_add_several(kvp_len, data):
    "Test filling up a KVP."
    ks = np.full(kvp_len, -1, dtype=np.int32)
    vs = np.zeros(kvp_len)

    n = 0

    values = np.random.randn(kvp_len) * 100

    kvp = KVPHeap(0, 0, kvp_len, ks, vs)
    for k, v in enumerate(values):
        n = kvp.insert(k, v)

    assert n == kvp_len
    # all key slots are used
    assert all(ks >= 0)
    # all keys are there
    assert all(np.sort(ks) == list(range(kvp_len)))
    # value is the smallest
    assert vs[0] == np.min(vs)

    # it rejects a smaller value; -10000 is below our min value
    special_k = 500
    n2 = kvp.insert(special_k, -10000)

    assert n2 == n
    assert all(ks != special_k)
    assert all(vs > -5000.0)

    # it inserts a larger value somewhere
    old_mk = ks[0]
    old_mv = vs[0]
    assume(np.median(vs) < 50)
    nv = data.draw(st.floats(np.median(vs), 100))
    n2 = kvp.insert(special_k, nv)

    assert n2 == n
    # the old value minimum key has been removed
    assert all(ks != old_mk)
    # the old minimum value has been removed
    assert all(vs > old_mv)
    assert np.count_nonzero(ks == special_k) == 1


@given(st.data())
def test_kvp_add_middle(data):
    "Test that KVP works in the middle of an array."
    ks = np.full(100, -1, dtype=np.int32)
    vs = np.full(100, np.nan)

    n = 25
    avs = []

    values = st.floats(-100, 100)
    kvp = KVPHeap(25, 25, 10, ks, vs)
    for k in range(25):
        v = data.draw(values)
        avs.append(v)
        n = kvp.insert(k, v)

    assert n == 35
    # all the keys
    assert all(ks[25:35] >= 0)
    # value is the smallest
    assert vs[25] == np.min(vs[25:35])
    # highest-ranked keys
    assert all(np.sort(vs[25:35]) == np.sort(avs)[15:])

    # early is untouched
    assert all(ks[:25] == -1)
    assert all(np.isnan(vs[:25]))
    assert all(ks[35:] == -1)
    assert all(np.isnan(vs[35:]))


def test_kvp_insert_min():
    ks = np.full(10, -1, dtype=np.int32)
    vs = np.zeros(10)

    n = 0

    # something less than existing data
    kvp = KVPHeap(0, 0, 10, ks, vs)
    n = kvp.insert(5, -3)
    assert n == 1
    assert ks[0] == 5
    assert vs[0] == -3.0

    # equal to existing data
    kvp = KVPHeap(0, 0, 10, ks, vs)
    n = kvp.insert(7, -3.0)
    assert n == 1
    assert ks[0] == 7
    assert vs[0] == -3.0

    # greater than to existing data
    kvp = KVPHeap(0, 0, 10, ks, vs)
    n = kvp.insert(9, 5.0)
    assert n == 1
    assert ks[0] == 9
    assert vs[0] == 5.0


@settings(deadline=None)
@given(nph.arrays(np.float64, 20, elements=st.floats(-100, 100), unique=True))
def test_kvp_sort(values):
    "Test that sorting logic works"
    ks = np.full(10, -1, dtype=np.int32)
    vs = np.zeros(10)

    n = 0

    kvp = KVPHeap(0, 0, 10, ks, vs)
    for k in range(20):
        v = values[k]
        n = kvp.insert(k, v)

    assert n == 10

    ovs = vs.copy()
    oks = ks.copy()
    ord = np.argsort(ovs)
    ord = ord[::-1]

    kvp.sort()
    assert vs[0] == np.max(ovs)
    assert vs[-1] == np.min(ovs)
    assert all(ks == oks[ord])
    assert all(vs == ovs[ord])
