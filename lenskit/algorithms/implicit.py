from collections import namedtuple
import warnings
import logging
import pandas as pd
import numpy as np

from ..matrix import sparse_ratings
from . import Trainable, Recommender

_logger = logging.getLogger(__name__)

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

        _logger.info('training %s on %s matrix (%d nnz)', self.algo_class, iur.shape, iur.nnz)

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

    def __getattr__(self, name):
        return self.__dict__['algo_kwargs'][name]

    def __getstate__(self):
        return (self.algo_class, self.algo_args, self.algo_kwargs)

    def __setstate__(self, rec):
        cls, args, kwargs = rec
        self.algo_class = cls
        self.algo_args = args
        self.algo_kwargs = kwargs

    def __str__(self):
        return 'Implicit({}, {}, {})'.format(self.algo_class.__name__, self.algo_args, self.algo_kwargs)


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
        from implicit.bpr import BayesianPersonalizedRanking
        super().__init__(BayesianPersonalizedRanking, *args, **kwargs)
