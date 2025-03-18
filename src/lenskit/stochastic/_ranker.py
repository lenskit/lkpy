from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import softmax

from lenskit.data import ItemList
from lenskit.data.query import QueryInput, RecQuery
from lenskit.pipeline import Component
from lenskit.random import DerivableSeed, RNGFactory, derivable_rng
from lenskit.stats import argtopn


@dataclass
class StochasticTopNConfig:
    """
    Configuration for :class:`StochasticTopNRanker`.
    """

    n: int | None = None
    """
    The number of items to select. -1 or ``None`` to return all scored items.
    """

    rng: DerivableSeed = None
    """
    Random number generator configuration.
    """

    transform: Literal["softmax", "linear"] | None = "softmax"
    """
    Transformation to convert scores into ranking probabilities.

    softmax
        Use the softmax of the item scores as the selection probabilities.
    linear
        Linearly re-scale item scores to be selection probabilities. This
        equivalent to min-max scaling the scores, then re-scaling to sum
        to 1.
    ``None``
        No transformation, except negative scores are clamped to (almost)
        zero.  Not recommended unless your item scorer emits multinomial
        probabilities.
    """
    scale: float = 1.0
    """
    Scalar multiplier to apply to scores prior to transformation.  This is
    equivalent to the :math:`\\beta` parameter for parameterized softmax
    transformation.  Larger values will decrease the entropy of the sampled
    rankings.
    """


class StochasticTopNRanker(Component[ItemList]):
    """
    Stochastic top-N ranking with optional weight transformation.

    This uses the exponential sampling method, a more efficient approximation of
    Plackett-Luce sampling than even the Gumbell trick, as documented by `Tim
    Vieira`_.  It expects a scored list of input items, and samples ``n`` items,
    with selection probabilities proportional to their scores.  Scores can be
    optionally rescaled (inverse temperature) and transformed (e.g. softmax).

    .. note::

        When no transformation is used, negative scores are still clamped to
        (approximately) zero.

    .. _`Tim Vieiera`: https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/

    Stability:
        Caller

    Args:
        n:
            The number of items to return (-1 to return unlimited).
        rng:
            The random number generator or specification (see :ref:`rng`).  This
            class supports derivable RNGs.
    """

    config: StochasticTopNConfig
    "Stochastic ranker configuration."

    _rng_factory: RNGFactory

    def __init__(self, config: StochasticTopNConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self._rng_factory = derivable_rng(self.config.rng)

    def __call__(
        self, items: ItemList, query: QueryInput | None = None, n: int | None = None
    ) -> ItemList:
        query = RecQuery.create(query)
        rng = self._rng_factory(query)

        scores = items.scores()
        if scores is None:
            raise ValueError("item list must have scores")

        valid_mask = np.isfinite(scores)
        valid_items = items[valid_mask]
        N = len(valid_items)
        if N == 0:
            return ItemList(item_ids=[], scores=[], ordered=True)

        if n is None or n < 0:
            n = self.config.n or -1

        if n < 0 or n > N:
            n = N

        # scale the scores — with softmax, this is the equivalent of β.
        # see: https://en.wikipedia.org/wiki/Softmax_function
        scores = scores[valid_mask] * self.config.scale
        match self.config.transform:
            case "linear":
                lb = np.min(scores).item()
                ub = np.max(scores).item()
                r = ub - lb
                # scale weights twice to reduce risk of floating-point error
                scores -= lb
                weights = None
                if r > 0:
                    scores /= r
                    tot = np.sum(scores)
                    if tot > 0:
                        weights = scores / tot

                if weights is None:
                    weights = np.ones_like(scores) / len(scores)
            case "softmax":
                weights = softmax(scores)
            case None:
                weights = scores

        # positive instead of negative, because we take top-N instead of bottom
        keys = np.log(rng.uniform(0, 1, N))
        # smoooth very small weights to avoid divide-by-zero
        keys /= np.maximum(weights, np.finfo("f4").smallest_normal)

        picked = argtopn(keys, n)
        return ItemList(valid_items[picked], ordered=True)
