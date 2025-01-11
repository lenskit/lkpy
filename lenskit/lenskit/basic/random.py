import numpy as np
from pydantic import BaseModel

from lenskit.data import ItemList
from lenskit.data.query import QueryInput, RecQuery
from lenskit.pipeline import Component
from lenskit.random import DerivableSeed, RNGFactory, derivable_rng
from lenskit.stats import argtopn


class RandomConfig(BaseModel, arbitrary_types_allowed=True):
    n: int | None = None
    """
    The number of items to select. -1 or ``None`` to return all scored items.
    """

    rng: DerivableSeed = None
    """
    Random number generator configuration.
    """


class RandomSelector(Component[ItemList]):
    """
    Randomly select items from a candidate list.

    Stability:
        Caller

    Args:
        n:
            The number of items to select, or ``-1`` to randomly permute the
            items.
        rng:
            The random number generator or specification (see :ref:`rng`).  This
            class supports derivable RNGs.
    """

    config: RandomConfig
    _rng_factory: RNGFactory

    def __init__(self, config: RandomConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self._rng_factory = derivable_rng(self.config.rng)

    def __call__(
        self, items: ItemList, query: QueryInput | None = None, n: int | None = None
    ) -> ItemList:
        """
        Args:
            items:
                The items from which to pick.
            query:
                The recommendation query; optional, and only consulted when the
                RNG seed is user-dependent.
            n:
                The number of items to select, overriding the configured value.
        """
        if n is None:
            n = self.config.n or -1

        query = RecQuery.create(query)
        rng = self._rng_factory(query)

        if n < 0:
            n = len(items)
        else:
            n = min(n, len(items))

        if n > 0:
            picked = rng.choice(len(items), n, replace=False)
            return items[picked]
        else:
            return items[np.zeros(0, dtype=np.int32)]


class SoftmaxRanker(Component[ItemList]):
    """
    Stochastic top-N ranking with softmax sampling.

    This uses the “softmax” sampling method, a more efficient approximation of
    Plackett-Luce sampling than even the Gumbell trick, as documented by `Tim
    Vieira`_.  It expects a scored list of input items, and samples ``n`` items,
    with selection probabilities proportional to their scores.

    .. note::

        Negative scores are clamped to (approximately) zero.

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

    config: RandomConfig
    _rng_factory: RNGFactory

    def __init__(self, config: RandomConfig | None = None, **kwargs):
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

        valid_mask = ~np.isnan(scores)
        valid_items = items[valid_mask]
        N = len(valid_items)
        if N == 0:
            return ItemList(item_ids=[], scores=[], ordered=True)

        if n is None or n < 0:
            n = self.config.n or -1

        if n < 0 or n > N:
            n = N

        keys = np.log(rng.uniform(0, 1, N))
        keys /= np.maximum(scores[valid_mask], 1.0e-10)

        picked = argtopn(keys, n)
        return ItemList(valid_items[picked], ordered=True)
