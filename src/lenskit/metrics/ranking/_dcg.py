# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Callable, TypeAlias, override

from lenskit.data import ItemList

from ._base import ListMetric, RankingMetricBase
from ._weighting import LogRankWeight, RankWeight

Discount: TypeAlias = Callable[[NDArray[np.number]], NDArray[np.float64]]


class DiscountWeight(RankWeight):
    def __init__(self, discount: Discount):
        self.discount = discount

    def weight(self, ranks):
        return np.reciprocal(np.maximum(self.discount(ranks), 1))


class NDCG(ListMetric, RankingMetricBase):
    """
    Compute the normalized discounted cumulative gain :cite:p:`ndcg`.

    Discounted cumultative gain is computed as:

    .. math::
        \\begin{align*}
        \\mathrm{DCG}(L,u) & = \\sum_{i=1}^{|L|} \\frac{r_{ui}}{d(i)}
        \\end{align*}

    Unrated items are assumed to have a utility of 0; if no rating values are
    provided in the truth frame, item ratings are assumed to be 1.

    This is then normalized as follows:

    .. math::
        \\begin{align*}
        \\mathrm{nDCG}(L, u) & = \\frac{\\mathrm{DCG}(L,u)}{\\mathrm{DCG}(L_{\\mathrm{ideal}}, u)}
        \\end{align*}

    Args:
        n:
            The maximum recommendation list length to consider (longer lists are
            truncated).
        weight:
            The rank weighting to use.
        discount:
            The discount function to use.  The default, base-2 logarithm, is the
            original function used by :cite:t:`ndcg`.  It is deprecated in favor
            of the ``weight`` option.
        gain:
            The field on the test data to use for gain values.  If ``None`` (the
            default), all items present in the test data have a gain of 1.  If
            set to a string, it is the name of a field (e.g. ``'rating'``).  In
            all cases, items not present in the truth data have a gain of 0.

    Stability:
        Caller
    """

    weight: RankWeight
    discount: Discount | None
    gain: str | None

    def __init__(
        self,
        n: int | None = None,
        *,
        k: int | None = None,
        weight: RankWeight = LogRankWeight(),
        discount: Discount | None = None,
        gain: str | None = None,
    ):
        super().__init__(n, k=k)
        self.weight = weight
        self.discount = discount
        if discount is not None:
            self.weight = DiscountWeight(discount)
        self.gain = gain

    @property
    def label(self):
        if self.n is not None:
            return f"NDCG@{self.n}"
        else:
            return "NDCG"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)

        if self.gain:
            realized = _graded_dcg(recs, test, self.gain, self.weight)

            gains = test.field(self.gain, "pandas", index="ids")
            if gains is None:
                raise KeyError(f"test items have no field {self.gain}")
            if self.n:
                gains = gains.nlargest(n=self.n)
            else:
                gains = gains.sort_values(ascending=False)
            iweight = self.weight.weight(np.arange(1, len(gains) + 1))
            ideal = np.dot(gains.values, iweight).item()  # type: ignore

        else:
            realized = _binary_dcg(recs, test, self.weight)
            n = len(test)
            if self.n and self.n < n:
                n = self.n
            ideal = fixed_dcg(n, self.weight)

        return realized / ideal


class DCG(ListMetric, RankingMetricBase):
    """
    Compute the _unnormalized_ discounted cumulative gain :cite:p:`ndcg`.

    Discounted cumultative gain is computed as:

    .. math::
        \\begin{align*}
        \\mathrm{DCG}(L,u) & = \\sum_{i=1}^{|L|} \\frac{r_{ui}}{d(i)}
        \\end{align*}

    Unrated items are assumed to have a utility of 0; if no rating values are
    provided in the truth frame, item ratings are assumed to be 1.

    This metric does *not* normalize by ideal DCG. For that, use :class:`NDCG`.
    See :cite:t:`jeunenNormalisedDiscountedCumulative2024` for an argument for
    using the unnormalized version.

    Args:
        n:
            The maximum recommendation list length to consider (longer lists are
            truncated).
        discount:
            The discount function to use.  The default, base-2 logarithm, is the
            original function used by :cite:t:`ndcg`.
        gain:
            The field on the test data to use for gain values.  If ``None`` (the
            default), all items present in the test data have a gain of 1.  If set
            to a string, it is the name of a field (e.g. ``'rating'``).  In all
            cases, items not present in the truth data have a gain of 0.

    Stability:
        Caller
    """

    weight: RankWeight
    discount: Discount | None
    gain: str | None

    def __init__(
        self,
        n: int | None = None,
        *,
        k: int | None = None,
        weight: RankWeight = LogRankWeight(),
        discount: Discount | None = None,
        gain: str | None = None,
    ):
        super().__init__(n, k=k)
        self.weight = weight
        self.discount = discount
        if discount is not None:
            self.weight = DiscountWeight(discount)
        self.gain = gain

    @property
    def label(self):
        if self.n is not None:
            return f"DCG@{self.n}"
        else:
            return "DCG"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)

        if self.gain:
            return _graded_dcg(recs, test, self.gain, self.weight)
        else:
            return _binary_dcg(recs, test, self.weight)


def _graded_dcg(
    recs: ItemList, test: ItemList, field: str, weight: RankWeight = LogRankWeight()
) -> float:
    gains = test.field(field, "pandas", index="ids")
    if gains is None:
        raise KeyError(f"test items have no field {field}")

    ranks = recs.ranks(format="pandas")
    if ranks is None:
        raise TypeError("item list is not ordered")

    ranks, gains = ranks.align(gains, join="inner")
    weights = weight.weight(ranks.values)  # type: ignore

    return np.dot(gains.values, weights)  # type: ignore


def _binary_dcg(recs: ItemList, test: ItemList, weight: RankWeight = LogRankWeight()) -> float:
    good = recs.isin(test)
    ranks = recs.ranks()
    if ranks is None:
        raise TypeError("item list is not ordered")

    weights = weight.weight(ranks[good])
    return np.sum(weights).item()


def array_dcg(
    scores: NDArray[np.number], weight: RankWeight = LogRankWeight(), *, graded: bool = True
):
    """
    Compute the Discounted Cumulative Gain of a series of recommended items with rating scores.
    These should be relevance scores; they can be :math:`{0,1}` for binary relevance data.

    This is not a true top-N metric, but is a utility function for other metrics.

    Args:
        scores:
            The utility scores of a list of recommendations, in recommendation order.
        discount:
            the rank discount function.  Each item's score will be divided the discount of its rank,
            if the discount is greater than 1.

    Returns:
        double: the DCG of the scored items.
    """
    ids = np.arange(1, len(scores) + 1)
    recs = ItemList(item_ids=ids, ordered=True)

    mask = scores > 0
    test = ItemList(item_ids=ids[mask], rating=scores[mask])

    if graded:
        return _graded_dcg(recs, test, "rating")
    else:
        return _binary_dcg(recs, test)


def fixed_dcg(n: int, weight: RankWeight = LogRankWeight()) -> float:
    """
    Compute the Discounted Cumulative Gain of a fixed number of items with
    relevance 1.
    """

    ranks = np.arange(1, n + 1)
    wvec = weight.weight(ranks)
    return np.sum(wvec).item()
