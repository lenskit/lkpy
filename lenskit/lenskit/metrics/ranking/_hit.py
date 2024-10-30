import numpy as np

from lenskit.data import ItemList

from ._base import RankingMetricBase


class Hit(RankingMetricBase):
    """
    Compute whether or not a list is a hit; any list with at least one
    relevant item in the first :math:`k` positions (:math:`L_{\\le k} \\cap
    I_u^{\\mathrm{test}} \\ne \\emptyset`) is scored as 1, and lists with no
    relevant items as 0.  When averaged over the recommendation lists, this
    computes the *hit rate* :cite:p:`deshpande:iknn`.
    """

    def __call__(self, recs: ItemList, test: ItemList) -> float:
        if len(test) == 0:
            return np.nan

        recs = self.truncate(recs)

        return 1 if np.any(np.isin(recs.ids(), test.ids())) else 0
