import numpy as np

from lenskit.data import ItemList

from ._base import RankingMetricBase


class recip_rank(RankingMetricBase):
    """
    Compute the reciprocal rank :cite:p:`trec5-confusion` of the first relevant
    item in a list of recommendations.  Taking the mean of this metric over the
    recommendation lists in a run yields the MRR (mean reciprocal rank).

    Let :math:`\\kappa` denote the 1-based rank of the first relevant item in
    :math:`L`, with :math:`\\kappa=\\infty` if none of the first :math:`k` items
    in :math:`L` are relevant; then the reciprocal rank is :math:`1 / \\kappa`.
    If no elements are relevant, the reciprocal rank is therefore 0.
    :cite:t:`deshpande:iknn` call this the “reciprocal hit rate”.
    """

    def __call__(self, recs: ItemList, test: ItemList) -> float:
        if len(test) == 0:
            return np.nan

        recs = self.truncate(recs)
        items = recs.ids()
        good = np.isin(items, test.ids())
        (npz,) = np.nonzero(good)
        if len(npz):
            return 1.0 / (npz[0] + 1.0)
        else:
            return 0.0
