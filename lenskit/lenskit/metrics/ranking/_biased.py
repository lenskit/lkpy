from ._base import RankingMetricBase


class RBP(RankingMetricBase):
    """
    Evaluate recommendations with rank-biased precision :cite:p:`rbp` with a
    patience parameter :math:`\\gamma`.

    If :math:`r_{ui} \\in \\{0, 1\\}` is binary implicit ratings, this is
    computed by:

    .. math::
        \\begin{align*}
        \\operatorname{RBP}_\\gamma(L, u) & =(1 - \\gamma) \\sum_i r_{ui} p^i
        \\end{align*}

    The original RBP metric depends on the idea that the rank-biased sum of
    binary relevance scores in an infinitely-long, perfectly-precise list has is
    :math:`1/(1 - \\gamma)`. However, in recommender evaluation, we usually have
    a small test set, so the maximum achievable RBP is significantly less, and
    is a function of the number of test items.  With ``normalize=True``, the RBP
    metric will be normalized by the maximum achievable with the provided test
    data.

    Args:
        recs:
            The recommendation list.  This is expected to have a column ``item``
            with the recommended item IDs; all other columns are ignored.
        truth:
            The user's test data. It is expected to be *indexed* by item ID. If
            it has a ``rating`` column, that is used as the item gains;
            otherwise, each item has gain 1. All other columns are ignored.
        k:
            The maximum recommendation list length.
        patience:
            The patience parameter :math:`\\gamma`, the probability that the
            user continues browsing at each point.
        normalize:
            Whether to normalize the RBP scores; if ``True``, divides the RBP
            score by the maximum achievable with the test data (as in nDCG).
    """

    pass
