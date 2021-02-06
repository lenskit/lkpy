import logging
import inspect
import pandas as pd
import numpy as np

from ..data import sparse_ratings
from . import Recommender, Predictor

_logger = logging.getLogger(__name__)


class BaseRec(Recommender, Predictor):
    """
    Base class for Implicit-backed recommenders.

    Args:
        delegate(implicit.RecommenderBase):
            The delegate algorithm.

    Attributes:
        delegate(implicit.RecommenderBase):
            The :py:mod:`implicit` delegate algorithm.
        matrix_(scipy.sparse.csr_matrix):
            The user-item rating matrix.
        user_index_(pandas.Index):
            The user index.
        item_index_(pandas.Index):
            The item index.
    """

    def __init__(self, delegate):
        self.delegate = delegate

    def fit(self, ratings, **kwargs):
        matrix, users, items = sparse_ratings(ratings, scipy=True)
        iur = matrix.T.tocsr()

        _logger.info('training %s on %s matrix (%d nnz)', self.delegate, iur.shape, iur.nnz)

        self.delegate.fit(iur)

        self.matrix_ = matrix
        self.user_index_ = users
        self.item_index_ = items

        return self

    def recommend(self, user, n=None, candidates=None, ratings=None):
        try:
            uid = self.user_index_.get_loc(user)
        except KeyError:
            return pd.DataFrame({'item': []})

        if candidates is None:
            i_n = n if n is not None else len(self.item_index_)
            recs = self.delegate.recommend(uid, self.matrix_, N=i_n)
        else:
            cands = self.item_index_.get_indexer(candidates)
            cands = cands[cands >= 0]
            recs = self.delegate.rank_items(uid, self.matrix_, cands)

        if n is not None:
            recs = recs[:n]
        rec_df = pd.DataFrame.from_records(recs, columns=['item_pos', 'score'])
        rec_df['item'] = self.item_index_[rec_df.item_pos]
        return rec_df.loc[:, ['item', 'score']]

    def predict_for_user(self, user, items, ratings=None):
        try:
            uid = self.user_index_.get_loc(user)
        except KeyError:
            return pd.Series(np.nan, index=items)

        iids = self.item_index_.get_indexer(items)
        iids = iids[iids >= 0]

        ifs = self.delegate.item_factors[iids]
        uf = self.delegate._user_factor(uid, None, False)
        scores = ifs.dot(uf)
        scores = pd.Series(scores, index=self.item_index_[iids])
        return scores.reindex(items)

    def __getattr__(self, name):
        if 'delegate' not in self.__dict__:
            raise AttributeError()
        dd = self.delegate.__dict__
        if name in dd:
            return dd[name]
        else:
            raise AttributeError()

    def get_params(self, deep=True):
        dd = self.delegate.__dict__
        sig = inspect.signature(self.delegate.__class__)
        names = list(sig.parameters.keys())
        return dict([(k, dd.get(k)) for k in names])

    def __str__(self):
        return 'Implicit({})'.format(self.delegate)


class ALS(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.als`.
    """
    def __init__(self, *args, **kwargs):
        """
        Construct an ALS recommender.  The arguments are passed as-is to
        :py:class:`implicit.als.AlternatingLeastSquares`.
        """
        from implicit.als import AlternatingLeastSquares
        super().__init__(AlternatingLeastSquares(*args, **kwargs))


class BPR(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.bpr`.
    """
    def __init__(self, *args, **kwargs):
        """
        Construct an ALS recommender.  The arguments are passed as-is to
        :py:class:`implicit.als.BayesianPersonalizedRanking`.
        """
        from implicit.bpr import BayesianPersonalizedRanking
        super().__init__(BayesianPersonalizedRanking(*args, **kwargs))
