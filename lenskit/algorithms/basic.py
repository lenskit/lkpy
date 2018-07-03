"""
Basic utility algorithms and combiners.
"""

import pandas as pd


class Memorized:
    """
    The memorized algorithm memorizes scores & repeats them.
    """

    def __init__(self, scores):
        """
        Args:
            scores(pandas.DataFrame): the scores to memorize.
        """
        self.scores = scores

    def train(self, ratings):
        return self.scores

    def predict(self, model, user, items, ratings=None):
        uscores = model[model.user == user]
        urates = uscores.set_index('item').rating
        return urates.reindex(items)


class Fallback:
    """
    The Fallback algorithm predicts with its first component, uses the second to fill in
    missing values, and so forth.
    """

    def __init__(self, *algorithms):
        """
        Args:
            algorithms: a list of component algorithms.
        """
        self.algorithms = algorithms

    def train(self, ratings):
        return [a.train(ratings) for a in self.algorithms]

    def predict(self, model, user, items, ratings=None):
        remaining = pd.Index(items)
        preds = None

        for algo, amod in zip(self.algorithms, model):
            aps = algo.predict(amod, user, remaining, ratings=ratings)
            aps = aps[aps.notna()]
            if preds is None:
                preds = aps
            else:
                preds = pd.concat([preds, aps])
            remaining = remaining.difference(preds.index)
            if len(remaining) == 0:
                break

        return preds.reindex(items)
