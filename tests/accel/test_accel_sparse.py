import numpy as np
from scipy.sparse import csr_array

from hypothesis import given

from lenskit._accel import make_csr
from lenskit.testing import sparse_arrays


def test_make_csr_empty():
    csr = make_csr(
        np.array([0], dtype=np.int64),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float32),
        (50, 100),
    )
    assert csr.nnz == 0
    assert csr.nrow == 50
    assert csr.ncol == 100


@given(sparse_arrays())
def test_make_csr(arr: csr_array):
    csr = make_csr(arr.indptr, np.require(arr.indices, "i4"), np.require(arr.data, "f4"), arr.shape)
    assert csr.nnz == arr.nnz
    assert csr.nrow == arr.shape[0]
    assert csr.ncol == arr.shape[1]

    assert np.all(csr.rowptrs == arr.indptr)
    assert np.all(csr.colinds == arr.indices)
    assert np.all(csr.values == arr.data.astype("f4"))
