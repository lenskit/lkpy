import numpy as np
from numpy.typing import NDArray
from typing_extensions import Callable, TypeAlias, override

from lenskit.data import ItemList

from ._base import ListMetric, RankingMetricBase

Discount: TypeAlias = Callable[[NDArray[np.number]], NDArray[np.float64]]


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
        k:
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

    discount: Discount
    gain: str | None

    def __init__(
        self, k: int | None = None, *, discount: Discount = np.log2, gain: str | None = None
    ):
        super().__init__(k=k)
        self.discount = discount
        self.gain = gain

    @property
    def label(self):
        if self.k is not None:
            return f"NDCG@{self.k}"
        else:
            return "NDCG"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)
        items = recs.ids()

        if self.gain:
            gains = test.field(self.gain, "pandas", index="ids")
            if gains is None:
                raise KeyError(f"test items have no field {self.gain}")
            scores = gains.reindex(items, fill_value=0).values
            if self.k:
                gains = gains.nlargest(n=self.k)
            else:
                gains = gains.sort_values(ascending=False)
            ideal = array_dcg(np.require(gains.values, np.float32), self.discount)
        else:
            scores = np.zeros_like(items, dtype=np.float32)
            scores[np.isin(items, test.ids())] = 1.0
            n = len(test)
            if self.k and self.k < n:
                n = self.k
            ideal = fixed_dcg(n, self.discount)

        realized = array_dcg(np.require(scores, np.float32), self.discount)
        return realized / ideal


def array_dcg(scores: NDArray[np.number], discount: Discount = np.log2):
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
    scores = np.nan_to_num(scores)
    ranks = np.arange(1, len(scores) + 1)
    disc = discount(ranks)
    np.maximum(disc, 1, out=disc)
    np.reciprocal(disc, out=disc)
    return np.dot(scores, disc)


def fixed_dcg(n: int, discount: Discount = np.log2):
    """
    Compute the Discounted Cumulative Gain of a fixed number of items with
    relevance 1.
    """

    ranks = np.arange(1, n + 1)
    disc = discount(ranks)
    disc = np.maximum(disc, 1)
    disc = np.reciprocal(disc)
    return np.sum(disc)
