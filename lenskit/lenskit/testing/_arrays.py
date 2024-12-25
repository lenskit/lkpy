"""
Hypothesis strategies that generate arrays and lists.
"""

from typing import Literal

import numpy as np
import scipy.sparse as sps

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import assume

from lenskit.data import ItemList
from lenskit.math.sparse import torch_sparse_from_scipy


@st.composite
def coo_arrays(
    draw,
    shape=None,
    dtype=nph.floating_dtypes(endianness="=", sizes=[32, 64]),
    elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False, width=32),
) -> sps.coo_array:
    if shape is None:
        shape = st.tuples(st.integers(1, 100), st.integers(1, 100))

    if isinstance(shape, st.SearchStrategy):
        shape = draw(shape)

    if not isinstance(shape, tuple):
        shape = shape, shape
    rows, cols = shape
    if isinstance(rows, st.SearchStrategy):
        rows = draw(rows)
    if isinstance(cols, st.SearchStrategy):
        cols = draw(cols)

    mask = draw(nph.arrays(np.bool_, (rows, cols)))
    # at least one nonzero value
    assume(np.any(mask))
    nnz = int(np.sum(mask))

    ris, cis = np.nonzero(mask)

    vs = draw(
        nph.arrays(dtype, nnz, elements=elements),
    )

    return sps.coo_array((vs, (ris, cis)), shape=(rows, cols))


@st.composite
def sparse_arrays(draw, *, layout="csr", **kwargs):
    if isinstance(layout, list):
        layout = st.sampled_from(layout)
    if isinstance(layout, st.SearchStrategy):
        layout = draw(layout)

    M: sps.coo_array = draw(coo_arrays(**kwargs))

    match layout:
        case "csr":
            return M.tocsr()
        case "csc":
            return M.tocsc()
        case "coo":
            return M
        case _:
            raise ValueError(f"invalid layout {layout}")


@st.composite
def sparse_tensors(draw, *, layout="csr", **kwargs):
    if isinstance(layout, list):
        layout = st.sampled_from(layout)
    if isinstance(layout, st.SearchStrategy):
        layout = draw(layout)

    M: sps.coo_array = draw(coo_arrays(**kwargs))
    return torch_sparse_from_scipy(M, layout)  # type: ignore


@st.composite
def scored_lists(
    draw: st.DrawFn,
    *,
    n: int | tuple[int, int] | st.SearchStrategy[int] = st.integers(0, 1000),
    scores: st.SearchStrategy[float] | Literal["gaussian"] | None = None,
) -> ItemList:
    """
    Hypothesis generator that produces scored lists.
    """
    if isinstance(n, st.SearchStrategy):
        n = draw(n)
    elif isinstance(n, tuple):
        n = draw(st.integers(*n))

    ids = np.arange(1, n + 1, dtype=np.int32)
    if scores == "gaussian":
        seed = draw(st.integers(0))
        rng = np.random.default_rng(seed)
        xs = np.exp(rng.normal(size=n))
    else:
        xs = draw(
            nph.arrays(nph.floating_dtypes(endianness="=", sizes=[32, 64]), n, elements=scores)
        )
    return ItemList(item_ids=ids, scores=xs)
