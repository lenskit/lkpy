"""
Algorithms to rank items based on scores.
"""

from . import Recommender, Predictor


class TopN(Recommender, Predictor):
    """
    Basic recommender that implements top-N recommendation using a predictor.

    .. note::
        This class does not do anything of its own in :meth:`fit`.  If its
        predictor and candidate selector are both fit, the top-N recommender
        does not need to be fit.

    Args:
        predictor(Predictor):
            The underlying predictor.
        selector(CandidateSelector):
            The candidate selector.  If ``None``, uses :class:`UnratedItemCandidateSelector`.
    """

    def __init__(self, predictor, selector=None):
        from .basic import UnratedItemCandidateSelector
        self.predictor = predictor
        self.selector = selector if selector is not None else UnratedItemCandidateSelector()

    def fit(self, ratings, **kwargs):
        """
        Fit the recommender.

        Args:
            ratings(pandas.DataFrame):
                The rating or interaction data.  Passed changed to the predictor and
                candidate selector.
            args, kwargs:
                Additional arguments for the predictor to use in its training process.
        """
        self.predictor.fit(ratings, **kwargs)
        self.selector.fit(ratings, **kwargs)
        return self

    def fit_iters(self, ratings, **kwargs):
        if not hasattr(self.predictor, 'fit_iters'):
            raise AttributeError('predictor has no method fit_iters')

        self.selector.fit(ratings, **kwargs)
        pred = self.predictor
        for p in pred.fit_iters(ratings, **kwargs):
            self.predictor = p
            yield self

        self.predictor = pred

    def recommend(self, user, n=None, candidates=None, ratings=None):
        if candidates is None:
            candidates = self.selector.candidates(user, ratings)

        scores = self.predictor.predict_for_user(user, candidates, ratings)
        scores = scores[scores.notna()]
        if n is not None:
            scores = scores.nlargest(n)
        else:
            scores = scores.sort_values(ascending=False)
        scores.name = 'score'
        scores.index.name = 'item'
        return scores.reset_index()

    def predict(self, pairs, ratings=None):
        return self.predictor.predict(pairs, ratings)

    def predict_for_user(self, user, items, ratings=None):
        return self.predictor.predict_for_user(user, items, ratings)

    def __str__(self):
        return 'TopN/' + str(self.predictor)
