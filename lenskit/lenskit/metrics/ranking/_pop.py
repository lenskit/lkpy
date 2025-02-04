from __future__ import annotations

import numpy as np
import pandas as pd
from typing_extensions import Literal, override

from lenskit.data.dataset import Dataset
from lenskit.data.items import ItemList

from ._base import ListMetric, RankingMetricBase


class MeanPopRank(ListMetric, RankingMetricBase):
    r"""
    Compute the _obscurity_ (mean popularity rank) of the recommendations.

    Unlike other metrics, this metric requires access to the training dataset in
    order to compute item popularity metrics.  Supply this as a constructor
    parameter.

    This metric represents the popularity rank as a quantile, based on the
    either the number of distinct users who have interacted with the item, or
    the total interactions (depending on the options — distinct users is the
    default).

    Let $q_i$ be the _popularity rank_, represented as a quantile, of item $i$.
    $q_i = 1$ for the most-popular item; $q_i=0$ for an item with no users or
    interactions (the quantiles are min-max scaled). This metric computes the
    mean of the quantile popularity ranks for the recommended items:

    .. math::
        \mathcal{M}(L) = \frac{1}{|L|} \sum_{i \in L} q_i

    This metric is based on the ``obscurity'' metric of
    :cite:t:`ekstrandSturgeonCoolKids2017` and the popularity-based item novelty
    metric of :cite:t:`vargasRankRelevanceNovelty2011`.

    Stability:
        Caller
    """

    item_ranks: pd.Series[float]

    def __init__(
        self, data: Dataset, k: int | None = None, count: Literal["users", "interactions"] = "users"
    ):
        super().__init__(k=k)
        stats = data.item_stats()
        match count:
            case "users":
                counts = stats["user_count"]
            case "interactions":
                counts = stats["count"]
            case _:  # pragma: nocover
                raise ValueError(f"invalid count {count}")

        # If we just computed quantiles, items with count 0 would have a small positive
        # count. Let's quantile the positive items, then add back in zeros.
        pos = counts[counts > 0]
        ranks = pos.rank(method="average", ascending=True)
        ranks /= len(pos)
        self.item_ranks = ranks.reindex(counts.index, fill_value=0)

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)
        nrecs = len(recs)
        if nrecs == 0:
            return np.nan

        items = recs.ids()
        ranks = self.item_ranks.reindex(items, fill_value=0)
        return ranks.mean()
