# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging

import numpy as np

from lenskit.data.items import ItemList
from lenskit.pipeline import Component
from lenskit.pipeline.types import Lazy

_logger = logging.getLogger(__name__)


class FallbackScorer(Component[ItemList]):
    """
    Scoring component that fills in missing scores using a fallback.

    Stability:
        Caller
    """

    config: None

    def __call__(self, primary: ItemList, backup: Lazy[ItemList]) -> ItemList:
        s = primary.scores()
        if s is None:
            return backup.get()

        s = s.copy()
        missing = np.isnan(s)
        if not np.any(missing):
            return primary

        bs = backup.get().scores("pandas", index="ids")
        if bs is None:
            return primary

        s[missing] = bs.reindex(primary.ids()[missing]).values
        return ItemList(primary, scores=s)
