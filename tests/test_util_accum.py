import numpy as np

from lenskit.util.accum import kvp_minheap_insert, kvp_minheap_sort


from hypothesis import given, assume, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph


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


@given(st.integers(10, 100), st.data())
def test_kvp_add_several(kvp_len, data):
    "Test filling up a KVP."
    ks = np.full(kvp_len, -1, dtype=np.int32)
    vs = np.zeros(kvp_len)

    n = 0

    values = st.floats(-1000, 1000)

    for k in range(kvp_len):
        v = data.draw(values)
        assume(v not in vs[:n])  # we can't keep drawing the same value
        n = kvp_minheap_insert(0, n, kvp_len, k, v, ks, vs)

    assert n == kvp_len
    # all key slots are used
    assert all(ks >= 0)
    # all keys are there
    assert all(np.sort(ks) == list(range(kvp_len)))
    # value is the smallest
    assert vs[0] == np.min(vs)

    # it rejects a smaller value; -10000 is below our min value
    special_k = 500
    n2 = kvp_minheap_insert(0, n, kvp_len, special_k, -10000.0, ks, vs)

    assert n2 == n
    assert all(ks != special_k)
    assert all(vs > -1000.0)

    # it inserts a larger value somewhere
    old_mk = ks[0]
    old_mv = vs[0]
    assume(np.median(vs) < 400)
    nv = data.draw(st.floats(np.median(vs), 500))
    n2 = kvp_minheap_insert(0, n, kvp_len, special_k, nv, ks, vs)

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
    for k in range(25):
        v = data.draw(values)
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


@settings(deadline=None)
@given(nph.arrays(np.float64, 20, elements=st.floats(-100, 100), unique=True))
def test_kvp_sort(values):
    "Test that sorting logic works"
    ks = np.full(10, -1, dtype=np.int32)
    vs = np.zeros(10)

    n = 0

    for k in range(20):
        v = values[k]
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
