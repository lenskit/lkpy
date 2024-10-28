import numpy as np

from lenskit.data import ItemList

from ._base import RankingMetric


class hit(RankingMetric):
    k: int | None = None

    def __init__(self, k: int | None = None):
        if k is not None and k < 0:
            raise ValueError("k must be positive or None")
        self.k = k

    def __call__(self, recs: ItemList, test: ItemList) -> float:
        """
        Compute whether or not a list is a hit; any list with at least one
        relevant item in the first :math:`k` positions (:math:`L_{\\le k} \\cap
        I_u^{\\mathrm{test}} \\ne \\emptyset`) is scored as 1, and lists with no
        relevant items as 0.  When averaged over the recommendation lists, this
        computes the *hit rate* :cite:p:`deshpande:iknn`.

        Args:
            recs:
                The recommendation list.  This is expected to have a column
                ``item`` with the recommended item IDs; all other columns are
                ignored.
            truth:
                The user's test data. It is expected to be *indexed* by item ID.
                If it has a ``rating`` column, that is used as the item gains;
                otherwise, each item has gain 1. All other columns are ignored.
            k:
                The maximum list length to consider.

        Returns:
            The hit value, or ``NaN`` if no relevant items are available.
        """

        if len(test) == 0:
            return np.nan

        if self.k is not None:
            if not recs.ordered:
                raise ValueError("top-k filtering requires ordered list")
            if len(recs) > self.k:
                recs = recs[: self.k]

        return 1 if np.any(np.isin(recs.ids(), test.ids())) else 0
