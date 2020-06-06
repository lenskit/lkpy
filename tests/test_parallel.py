import multiprocessing as mp
import numpy as np

from lenskit.util.parallel import invoker, proc_count
from lenskit.util.test import set_env_var

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


def test_proc_count_default():
    with set_env_var('LK_NUM_PROCS', None):
        assert proc_count() == mp.cpu_count() // 2
        assert proc_count(level=1) == 2


def test_proc_count_no_div():
    with set_env_var('LK_NUM_PROCS', None):
        assert proc_count(1) == mp.cpu_count()


def test_proc_count_env():
    with set_env_var('LK_NUM_PROCS', '17'):
        assert proc_count() == 17
        assert proc_count(level=1) == 1


def test_proc_count_max():
    with set_env_var('LK_NUM_PROCS', None):
        assert proc_count(max_default=1) == 1


def test_proc_count_nest_env():
    with set_env_var('LK_NUM_PROCS', '7,3'):
        assert proc_count() == 7
        assert proc_count(level=1) == 3
        assert proc_count(level=2) == 1
