import numpy as np

from lenskit.util import Accumulator


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
