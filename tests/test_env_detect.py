import numba

import lenskit.util.debug as d
from lenskit.util.test import wantjit


def test_numba_info():
    ni = d.numba_info()
    if numba.config.DISABLE_JIT:
        assert ni is None
    else:
        assert ni is not None
        assert ni.threading == numba.threading_layer()
        assert ni.threads == numba.get_num_threads()


def test_blas_info():
    bi = d.blas_info()
    assert bi is not None
    # we can't really do more advanced tests
