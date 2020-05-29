"""
Numba-accleratable matrix operations.
"""

import logging
import numpy as np
import numba as n
from numba import njit, objmode
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass

from ..util.array import swap
from . import _CSR

_log = logging.getLogger(__name__)

_CSR64 = type('_CSR64', _CSR.__bases__, dict(_CSR.__dict__))
_CSR = jitclass({
    'nrows': n.intc,
    'ncols': n.intc,
    'nnz': n.intc,
    'rowptrs': n.intc[::1],
    'colinds': n.intc[::1],
    'values': n.float64[::1]
})(_CSR)
_CSR64 = jitclass({
    'nrows': n.intc,
    'ncols': n.intc,
    'nnz': n.int64,
    'rowptrs': n.int64[::1],
    'colinds': n.intc[::1],
    'values': n.float64[::1]
})(_CSR64)


@njit(nogil=True)
def _center_rows(csr: _CSR):
    means = np.zeros(csr.nrows)
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        if sp == ep:
            continue  # empty row
        vs = csr.row_vs(i)
        m = np.mean(vs)
        means[i] = m
        csr.values[sp:ep] -= m

    return means


@njit(nogil=True)
def _unit_rows(csr: _CSR):
    norms = np.zeros(csr.nrows)
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        if sp == ep:
            continue  # empty row
        vs = csr.row_vs(i)
        m = np.linalg.norm(vs)
        norms[i] = m
        csr.values[sp:ep] /= m

    return norms


@njit(nogil=True)
def _csr_align(rowinds, nrows, rowptrs, align):
    rcts = np.zeros(nrows, dtype=rowptrs.dtype)
    for r in rowinds:
        rcts[r] += 1

    rowptrs[1:] = np.cumsum(rcts)
    rpos = rowptrs[:-1].copy()

    for i in range(len(rowinds)):
        row = rowinds[i]
        pos = rpos[row]
        align[pos] = i
        rpos[row] += 1


@njit(nogil=True)
def _csr_align_inplace(shape, rows, cols, vals):
    """
    Align COO data in-place for a CSR matrix.

    Args:
        shape: the matrix shape
        rows: the matrix row indices (not modified)
        cols: the matrix column indices (**modified**)
        vals: the matrix values (**modified**)

    Returns:
        the CSR row pointers
    """
    nrows, ncols = shape
    nnz = len(rows)

    with objmode():
        _log.debug('aligning matrix with shape (%d, %d) and %d nnz', nrows, ncols, nnz)

    rps = np.zeros(nrows + 1, np.int64)

    for i in range(nnz):
        rps[rows[i] + 1] += 1
    for i in range(nrows):
        rps[i+1] += rps[i]

    rci = rps[:nrows].copy()

    with objmode():
        _log.debug('counted row sizes (largest %d), beginning shuffle', np.max(np.diff(rps)))

    pos = 0
    row = 0
    rend = rps[1]

    # skip to first nonempty row
    while row < nrows and rend == 0:
        row += 1
        rend = rps[row + 1]

    while pos < nnz:
        r = rows[pos]
        # swap until we have something in place
        while r != row:
            tgt = rci[r]
            # swap with the target position
            swap(cols, pos, tgt)
            if vals is not None:
                swap(vals, pos, tgt)

            # update the target start pointer
            rci[r] += 1

            # update the loop check
            r = rows[tgt]

        # now the current entry in the arrays is good
        # we need to advance to the next entry
        pos += 1
        rci[row] += 1

        # skip finished rows
        while pos == rend and pos < nnz:
            row += 1
            pos = rci[row]
            rend = rps[row+1]

    return rps


@njit
def _empty_csr(nrows, ncols, sizes):
    nnz = np.sum(sizes)
    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    for i in range(nrows):
        rowptrs[i+1] = rowptrs[i] + sizes[i]
    colinds = np.full(nnz, -1, dtype=np.intc)
    values = np.full(nnz, np.nan)
    return _CSR(nrows, ncols, nnz, rowptrs, colinds, values)


@njit
def _subset_rows(csr, begin, end):
    st = csr.rowptrs[begin]
    ed = csr.rowptrs[end]
    rps = csr.rowptrs[begin:(end+1)] - st

    cis = csr.colinds[st:ed]
    if csr.values.size == 0:
        vs = csr.values
    else:
        vs = csr.values[st:ed]
    return _CSR(end - begin, csr.ncols, ed - st, rps, cis, vs)
