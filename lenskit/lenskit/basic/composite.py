from __future__ import annotations

import logging

import numpy as np

from lenskit.data.items import ItemList
from lenskit.pipeline import Component

_logger = logging.getLogger(__name__)


class FallbackScorer(Component):
    """
    Scoring component that fills in missing scores using a fallback.

    Stability:
        Caller
    """

    config: None

    def __call__(self, scores: ItemList, backup: ItemList) -> ItemList:
        s = scores.scores()
        if s is None:
            return backup

        s = s.copy()
        missing = np.isnan(s)
        if not np.any(missing):
            return scores

        bs = backup.scores("pandas", index="ids")
        if bs is None:
            return scores

        s[missing] = bs.reindex(scores.ids()[missing]).values
        return ItemList(scores, scores=s)
