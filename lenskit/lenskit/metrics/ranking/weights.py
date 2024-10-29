"""
Weights or discounts for rank effectiveness metrics.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class RankWeights(Protocol):
    """
    Inferface for rank weight/discount implementations.

    This can be implemented directly as a function (whose signature matches
    :meth:`__call__`), or with a class object (for configurable weight models).
    """

    def __call__(
        self,
        n: int,
        *,
        scores: NDArray[np.number] | None = None,
        utilities: NDArray[np.number] | None = None,
    ) -> NDArray[np.floating]:
        """
        Compute the weights for a ranked set of items (general interface).

        Args:
            n:
                The number of rank positions for which to compute discounts.
            scores:
                The item scores, as computed by the system (optional, most
                discounts won't use this).
            utilities:
                The item ground-truth uiltity or relevance scores (usually 0/1
                indicators or ratings). Most discounts also don't use this, but
                relevance-dependent cascade models may use it.

        Returns:
            An array of discounts for ranks :math:`[1,n]`.
        """
        ...


class LogWeights(RankWeights):
    """
    Logarithmic weighting (as used in NDCG's original formulation
    :cite:p:`ndcg`).

    If the log of rank is less than 1, it is clamped to 1 (as in the original
    NDCG paper).  An alternative approach to handle rank 1 is to add 1 to all
    ranks, this can be achieved with the ``offset`` option (default is 0,
    matching the original paper :cite:p:`ndcg`).

    Args:
        base:
            The base of the logarithm.
        offset:
            An offset to add to ranks to deal with :math:`\\ln 1 = 0`.
    """

    base: float
    offset: int
    _cache: NDArray[np.float64] | None = None

    def __init__(self, base: float = 2, offset: int = 0):
        self.base = base
        self.offset = offset

    def __call__(self, n: int, **kwargs) -> NDArray[np.float64]:
        if self._cache is not None and len(self._cache) >= n:
            return self._cache[:n]

        ranks = np.arange(1 + self.offset, n + 1 + self.offset)
        if self.base == 2:
            weights = np.log2(ranks)
        elif self.base == 10:
            weights = np.log10(ranks)
        else:
            weights = np.log(ranks) / np.log(self.base)

        weights[weights < 1] = 1
        self._cache = weights
        return weights
