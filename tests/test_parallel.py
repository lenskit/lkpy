import numpy as np

from lenskit.util.parallel import invoker

from pytest import mark


def _mul_op(m, v):
    return m @ v


@mark.parametrize('n_jobs', [None, 1, 2, 8])
def test_invoke_matrix(n_jobs):
    matrix = np.random.randn(100, 100)
    vectors = [np.random.randn(100) for i in range(100)]
    with invoker(matrix, _mul_op, n_jobs) as inv:
        mults = inv.map(vectors)
        for rv, v in zip(mults, vectors):
            act_rv = matrix @ v
            assert np.all(rv == act_rv)
