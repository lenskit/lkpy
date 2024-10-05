import numpy as np

from lenskit.data import ItemList
from lenskit.data.query import QueryInput, RecQuery
from lenskit.pipeline import Component
from lenskit.util.random import DerivableSeed, RNGFactory, derivable_rng


class RandomSelector(Component):
    """
    Randomly select items from a candidate list.

    Args:
        n:
            The number of items to select, or ``-1`` to randomly permute the
            items.
        rng_spec:
            Seed or random state for generating recommendations.  Pass
            ``'user'`` to deterministically derive per-user RNGS from the user
            IDs for reproducibility.
    """

    IGNORED_CONFIG_FIELDS = ["rng_spec"]

    n: int
    _rng_factory: RNGFactory

    def __init__(self, n: int = -1, rng_spec: DerivableSeed = None):
        self.n = n
        self._rng_factory = derivable_rng(rng_spec)

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
            n = self.n

        query = RecQuery.create(query)
        keys = [query.user_id] if query.user_id else []
        rng = self._rng_factory(*keys)

        if n < 0:
            n = len(items)
        else:
            n = min(n, len(items))

        if n > 0:
            picked = rng.choice(len(items), n, replace=False)
            return items[picked]
        else:
            return items[np.zeros(0, dtype=np.int32)]
