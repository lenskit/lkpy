import numpy as np

from lenskit.util.accum import kvp_minheap_insert, kvp_minheap_sort
from lenskit.util.test import repeated


def test_kvp_add_to_empty():
    ks = np.empty(10, dtype=np.int32)
    vs = np.empty(10)

    # insert an item
    n = kvp_minheap_insert(0, 0, 10, 5, 3.0, ks, vs)

    # ep has moved
    assert n == 1

    # item is there
    assert ks[0] == 5
    assert vs[0] == 3.0


def test_kvp_add_larger():
    ks = np.empty(10, dtype=np.int32)
    vs = np.empty(10)

    # insert an item
    n = kvp_minheap_insert(0, 0, 10, 5, 3.0, ks, vs)
    n = kvp_minheap_insert(0, n, 10, 1, 6.0, ks, vs)

    # ep has moved
    assert n == 2

    # data is there
    assert all(ks[:2] == [5, 1])
    assert all(vs[:2] == [3.0, 6.0])


def test_kvp_add_smaller():
    ks = np.empty(10, dtype=np.int32)
    vs = np.empty(10)

    # insert an item
    n = kvp_minheap_insert(0, 0, 10, 5, 3.0, ks, vs)
    n = kvp_minheap_insert(0, n, 10, 1, 1.0, ks, vs)

    # ep has moved
    assert n == 2

    # data is there
    assert all(ks[:2] == [1, 5])
    assert all(vs[:2] == [1.0, 3.0])


@repeated(100)
def test_kvp_add_several():
    kvp_len = 50
    ks = np.full(kvp_len, -1, dtype=np.int32)
    vs = np.zeros(kvp_len)

    n = 0

    for k in range(kvp_len):
        v = np.random.randn()
        n = kvp_minheap_insert(0, n, kvp_len, k, v, ks, vs)

    assert n == kvp_len
    # all key slots are used
    assert all(ks >= 0)
    # all keys are there
    assert all(np.sort(ks) == list(range(kvp_len)))
    # value is the smallest
    assert vs[0] == np.min(vs)

    # it rejects a smaller value; -10000 is extremely unlikely
    special_k = 500
    n2 = kvp_minheap_insert(0, n, kvp_len, special_k, -10000.0, ks, vs)

    assert n2 == n
    assert all(ks != special_k)
    assert all(vs > -100.0)

    # it inserts a larger value somewhere; all positive is extremely unlikely
    old_mk = ks[0]
    old_mv = vs[0]
    n2 = kvp_minheap_insert(0, n, kvp_len, special_k, 0.0, ks, vs)

    assert n2 == n
    # the old value minimum key has been removed
    assert all(ks != old_mk)
    # the old minimum value has been removed
    assert all(vs > old_mv)
    assert np.count_nonzero(ks == special_k) == 1


@repeated
def test_kvp_add_middle():
    ks = np.full(100, -1, dtype=np.int32)
    vs = np.full(100, np.nan)

    n = 25
    avs = []

    for k in range(25):
        v = np.random.randn()
        avs.append(v)
        n = kvp_minheap_insert(25, n, 10, k, v, ks, vs)

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
    n = kvp_minheap_insert(0, n, 10, 5, -3.0, ks, vs)
    assert n == 1
    assert ks[0] == 5
    assert vs[0] == -3.0

    # equal to existing data
    n = kvp_minheap_insert(0, 0, 10, 7, -3.0, ks, vs)
    assert n == 1
    assert ks[0] == 7
    assert vs[0] == -3.0

    # greater than to existing data
    n = kvp_minheap_insert(0, 0, 10, 9, 5.0, ks, vs)
    assert n == 1
    assert ks[0] == 9
    assert vs[0] == 5.0


@repeated
def test_kvp_sort():
    ks = np.full(10, -1, dtype=np.int32)
    vs = np.zeros(10)

    n = 0

    for k in range(20):
        v = np.random.randn()
        n = kvp_minheap_insert(0, n, 10, k, v, ks, vs)

    assert n == 10

    ovs = vs.copy()
    oks = ks.copy()
    ord = np.argsort(ovs)
    ord = ord[::-1]

    kvp_minheap_sort(0, n, ks, vs)
    assert vs[0] == np.max(ovs)
    assert vs[-1] == np.min(ovs)
    assert all(ks == oks[ord])
    assert all(vs == ovs[ord])
