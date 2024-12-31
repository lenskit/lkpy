import numpy as np
from typing_extensions import override

from lenskit.data import ItemList

from ._base import ListMetric, RankingMetricBase


class Hit(ListMetric, RankingMetricBase):
    """
    Compute whether or not a list is a hit; any list with at least one
    relevant item in the first :math:`k` positions (:math:`L_{\\le k} \\cap
    I_u^{\\mathrm{test}} \\ne \\emptyset`) is scored as 1, and lists with no
    relevant items as 0.  When averaged over the recommendation lists, this
    computes the *hit rate* :cite:p:`deshpande:iknn`.

    Stability:
        Caller
    """

    @property
    def label(self):
        if self.k is not None:
            return f"Hit@{self.k}"
        else:
            return "Hit"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        if len(test) == 0:
            return np.nan

        recs = self.truncate(recs)

        return 1 if np.any(np.isin(recs.ids(), test.ids())) else 0
