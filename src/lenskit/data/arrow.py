"""
Utility functions for working with Arrow data.

.. stability:: internal
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from lenskit.logging import get_logger

_log = get_logger(__name__)


def is_sorted(table: pa.Table, key: list[str]) -> bool:
    assert len(key) >= 1, "must have at least one key column"
    log = _log.bind(nrows=table.num_rows)

    k1 = key[0]
    log.debug("checking first column", column=k1)
    d1 = pc.pairwise_diff(table.column(k1))
    if pc.any(pc.less(d1, 0)).as_py():
        log.debug("initial column non-monotonic", column=k1)
        return False

    changes = pc.greater(d1, 0).to_numpy()
    for k in key:
        log.debug("checking column", column=k)
        dk = pc.pairwise_diff(table.column(k)).to_numpy()
        if np.any(~changes & (dk < 0)):
            log.debug("column has disallowed reset", column=k)
            return False
        # later chnage allowed when either changes
        changes |= dk > 0

    return True
