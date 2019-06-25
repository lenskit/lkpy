# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:09:10 2019

@author: mohamed.cherif
"""

import logging
import pandas as pd
import numpy as np
import inspect

from ..matrix import ratings_surprise
from . import Recommender

_logger = logging.getLogger(__name__)



class SurpriseBaseRec(Recommender):
    """
    Base class for surprise-backed recommenders.
    Args:
        delegate(implicit.RecommenderBase):
            The delegate algorithm.
    Attributes:
        delegate(implicit.RecommenderBase):
            The :py:mod:`surprise` delegate algorithm.
        matrix_(scipy.sparse.csr_matrix):
            The user-item rating matrix.
        user_index_(pandas.Index):
            The user index.
        item_index_(pandas.Index):
            The item index.
    """

    def __init__(self, delegate):
        self.delegate = delegate

    def fit(self, ratings):
        dataset = ratings_surprise(ratings)
        trainset = dataset.build_full_trainset()

        _logger.info('training %s ', self.delegate)

        self.delegate.fit(trainset)
        
        self.ratings = ratings
        self.trainset = trainset
        self.inner_user_index_ = trainset.all_users()
        self.inner_item_index_ = trainset.all_items()

        return self

    def recommend(self, user, n=None, ratings=None):
        
        ##Getting user inner id
        inner_uid = self.trainset.to_inner_uid(user)
        # getting all items
        all_items = self.ratings["item"].unique()
        # getting already rated items
        rated_items = self.ratings.loc[self.ratings["user"] == user, "item"]
        # Remove items that user has rated
        items_to_pred = np.setdiff1d(all_items,rated_items)
        # Converting Item to item_innerId
        items_to_pred_converted = [self.trainset.to_inner_iid(item_to_pred) for item_to_pred in items_to_pred]        
        
        estimations = [self.delegate.estimate(inner_uid,item_id) for item_id in items_to_pred_converted]
        
        predictions_df = pd.DataFrame()
        predictions_df["item"] = items_to_pred
        predictions_df["prediction"] = estimations
        
        # Getting sorted prediction for a user
        sorted_items = predictions_df.sort_values(["prediction"],ascending=False)
        
        # selecting top n items        
        if n is not None:
            sorted_items = sorted_items.head(n)
        return sorted_items.loc[:, ['item', 'prediction']]

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
        return 'Surprise({})'.format(self.delegate)


