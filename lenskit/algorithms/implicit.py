from collections import namedtuple
import pandas as pd
import numpy as np

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

from ..matrix import sparse_ratings
from . import Trainable, Recommender

ImplicitModel = namedtuple('ImplicitModel', [
    'algo', 'matrix', 'users', 'items'
])
ImplicitModel.__doc__ = '''
Model for *implicit*-backed recommenders.

Attributes:
    algo(implicit.RecommenderBase): the underlying algorithm.
    matrix(scipy.sparse.csr_matrix): the user-item matrix.
    users(pandas.Index): the user ID to user position index.
    items(pandas.Index): the item ID to item position index.
'''


class BaseRec(Trainable, Recommender):
    """
    Base class for Implicit-backed recommenders.
    """
    def __init__(self, algo, *args, **kwargs):
        self.algo_class = algo
        self.algo_args = args
        self.algo_kwargs = kwargs

    def train(self, ratings):
        matrix, users, items = sparse_ratings(ratings, scipy=True)
        iur = matrix.T.tocsr()

        algo = self.algo_class(*self.algo_args, **self.algo_kwargs)
        algo.fit(iur)

        return ImplicitModel(algo, matrix, users, items)

    def recommend(self, model: ImplicitModel, user, n=None, candidates=None, ratings=None):
        try:
            uid = model.users.get_loc(user)
        except KeyError:
            return pd.DataFrame({'item': []})

        if candidates is None:
            recs = model.algo.recommend(uid, model.matrix, N=n)
        else:
            cands = model.items.get_indexer(candidates)
            cands = cands[cands >= 0]
            recs = model.algo.rank_items(uid, model.matrix, cands)

        if n is not None:
            recs = recs[:n]
        rec_df = pd.DataFrame.from_records(recs, columns=['item_pos', 'score'])
        rec_df['item'] = model.items[rec_df.item_pos]
        return rec_df.loc[:, ['item', 'score']]


class ALS(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.als`.
    """
    def __init__(self, *args, **kwargs):
        """
        Construct an ALS recommender.  The arguments are passed as-is to
        :py:class:`implicit.als.AlternatingLeastSquares`.
        """
        super().__init__(AlternatingLeastSquares, *args, **kwargs)


class BPR(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.bpr`.
    """
    def __init__(self, *args, **kwargs):
        """
        Construct an ALS recommender.  The arguments are passed as-is to
        :py:class:`implicit.als.BayesianPersonalizedRanking`.
        """
        super().__init__(BayesianPersonalizedRanking, *args, **kwargs)
