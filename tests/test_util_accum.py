import numpy as np

from lenskit.util import Accumulator
from lenskit.util.accum import kvp_insert


def test_accum_init_empty():
    values = np.empty(0)
    acc = Accumulator(values, 10)

    assert acc is not None
    assert acc.size == 0
    assert acc.peek() < 0
    assert acc.remove() < 0
    assert len(acc.top_keys()) == 0


def test_accum_add_get():
    values = np.array([1.5])
    acc = Accumulator(values, 10)

    assert acc is not None
    assert acc.size == 0
    assert acc.peek() < 0
    assert acc.remove() < 0

    acc.add(0)
    assert acc.size == 1
    assert acc.peek() == 0
    assert acc.remove() == 0
    assert acc.size == 0
    assert acc.peek() == -1


def test_accum_add_a_few():
    values = np.array([1.5, 2, -1])
    acc = Accumulator(values, 10)

    assert acc is not None
    assert acc.size == 0

    acc.add(1)
    acc.add(0)
    acc.add(2)

    assert acc.size == 3
    assert acc.peek() == 2
    assert acc.remove() == 2
    assert acc.size == 2
    assert acc.remove() == 0
    assert acc.remove() == 1
    assert acc.size == 0


def test_accum_add_a_few_lim():
    values = np.array([1.5, 2, -1])
    acc = Accumulator(values, 2)

    assert acc is not None
    assert acc.size == 0

    acc.add(1)
    acc.add(0)
    acc.add(2)

    assert acc.size == 2
    assert acc.remove() == 0
    assert acc.size == 1
    assert acc.remove() == 1
    assert acc.size == 0


def test_accum_add_more_lim():
    for run in range(10):
        values = np.random.randn(100)
        acc = Accumulator(values, 10)

        order = np.arange(len(values), dtype=np.int_)
        np.random.shuffle(order)
        for i in order:
            acc.add(i)
            assert acc.size <= 10

        topn = []
        # start with the smallest remaining one, grab!
        while acc.size > 0:
            topn.append(acc.remove())

        topn = np.array(topn)
        xs = np.argsort(values)
        assert all(topn == xs[-10:])


def test_accum_top_indices():
    for run in range(10):
        values = np.random.randn(100)
        acc = Accumulator(values, 10)

        order = np.arange(len(values), dtype=np.int_)
        np.random.shuffle(order)
        for i in order:
            acc.add(i)
            assert acc.size <= 10

        topn = acc.top_keys()

        xs = np.argsort(values)
        # should be top N values in decreasing order
        assert all(topn == np.flip(xs[-10:], axis=0))


def test_kvp_add_to_empty():
    ks = np.empty(10, dtype=np.int32)
    vs = np.empty(10)

    # insert an item
    n = kvp_insert(0, 0, 10, 5, 3.0, ks, vs)

    # ep has moved
    assert n == 1

    # item is there
    assert ks[0] == 5
    assert vs[0] == 3.0


def test_kvp_add_larger():
    ks = np.empty(10, dtype=np.int32)
    vs = np.empty(10)

    # insert an item
    n = kvp_insert(0, 0, 10, 5, 3.0, ks, vs)
    n = kvp_insert(0, n, 10, 1, 6.0, ks, vs)

    # ep has moved
    assert n == 2

    # data is there
    assert all(ks[:2] == [5, 1])
    assert all(vs[:2] == [3.0, 6.0])


def test_kvp_add_smaller():
    ks = np.empty(10, dtype=np.int32)
    vs = np.empty(10)

    # insert an item
    n = kvp_insert(0, 0, 10, 5, 3.0, ks, vs)
    n = kvp_insert(0, n, 10, 1, 1.0, ks, vs)

    # ep has moved
    assert n == 2

    # data is there
    assert all(ks[:2] == [1, 5])
    assert all(vs[:2] == [1.0, 3.0])


def test_kvp_add_several():
    ks = np.full(10, -1, dtype=np.int32)
    vs = np.zeros(10)

    n = 0

    for k in range(10):
        v = np.random.randn()
        n = kvp_insert(0, n, 10, k, v, ks, vs)

    assert n == 10
    # all the keys
    assert all(ks >= 0)
    assert all(np.sort(ks) == list(range(10)))
    # value is the smallest
    assert vs[0] == np.min(vs)

    # it rejects a smaller value; -100 is extremely unlikely
    n2 = kvp_insert(0, n, 10, 50, -100.0, ks, vs)

    assert n2 == n
    assert all(ks != 50)
    assert all(vs > -100.0)

    # it inserts a larger value; all positive is extremely unlikely
    old_mk = ks[0]
    old_mv = vs[0]
    n2 = kvp_insert(0, n, 10, 50, 0.0, ks, vs)

    assert n2 == n
    assert all(ks != old_mk)
    assert all(vs > old_mv)
    assert np.count_nonzero(ks == 50) == 1
