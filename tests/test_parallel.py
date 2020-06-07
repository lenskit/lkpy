import logging
import multiprocessing as mp
import numpy as np

from lenskit.util.parallel import invoker, proc_count, run_sp
from lenskit.util.test import set_env_var
from lenskit.sharing import persist

from pytest import mark, raises

_log = logging.getLogger(__name__)


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


def _sp_matmul(a1, a2, *, fail=False):
    _log.info('in worker process')
    if fail:
        raise RuntimeError('you rang?')
    else:
        return a1 @ a2


def _sp_matmul_p(a1, a2, *, fail=False):
    _log.info('in worker process')
    return persist(a1 @ a2).transfer()


def test_run_sp():
    a1 = np.random.randn(100, 100)
    a2 = np.random.randn(100, 100)

    res = run_sp(_sp_matmul, a1, a2)
    assert np.all(res == a1 @ a2)


def test_run_sp_fail():
    a1 = np.random.randn(100, 100)
    a2 = np.random.randn(100, 100)

    with raises(ChildProcessError):
        run_sp(_sp_matmul, a1, a2, fail=True)


def test_run_sp_persist():
    a1 = np.random.randn(100, 100)
    a2 = np.random.randn(100, 100)

    res = run_sp(_sp_matmul_p, a1, a2)
    try:
        assert res.is_owner
        assert np.all(res.get() == a1 @ a2)
    finally:
        res.close()
