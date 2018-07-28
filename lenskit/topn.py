import pandas as pd
import numpy as np

from .metrics.topn import *


class UnratedCandidates:
    """
    Candidate selector that selects unrated items from a training set.

    Args:
        training(pandas.DataFrame):
            the training data; must have ``user`` and ``item`` columns.
    """

    def __init__(self, training):
        self.training = training.set_index('user').item
        self.items = training.item.unique()

    def __call__(self, user):
        urates = self.training.loc[user]
        return np.setdiff1d(self.items, urates)
