# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
from pydantic import BaseModel

from lenskit.data import ItemList
from lenskit.data.query import QueryInput, RecQuery
from lenskit.pipeline import Component
from lenskit.random import DerivableSeed, RNGFactory, derivable_rng


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
