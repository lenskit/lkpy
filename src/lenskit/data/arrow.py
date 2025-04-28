# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Utility functions for working with Arrow data.

.. stability:: internal
"""

import numpy as np
import pyarrow as pa

from lenskit.logging import get_logger

_log = get_logger(__name__)


def is_sorted(table: pa.Table, key: list[str]) -> bool:
    assert len(key) >= 1, "must have at least one key column"
    log = _log.bind(nrows=table.num_rows)

    k1 = key[0]
    log.debug("checking first column", column=k1)
    d1 = np.diff(table.column(k1).to_numpy())
    if np.any(d1 < 0):
        log.debug("initial column non-monotonic", column=k1)
        return False

    changes = d1 > 0
    for k in key:
        log.debug("checking column", column=k)
        dk = np.diff(table.column(k).to_numpy())
        if np.any(~changes & (dk < 0)):
            log.debug("column has disallowed reset", column=k)
            return False
        # later chnage allowed when either changes
        changes |= dk > 0

    return True
