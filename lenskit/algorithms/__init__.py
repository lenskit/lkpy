"""
LensKit algorithms.

The `lenskit.algorithms` package contains several example algorithms for carrying out recommender
experiments.  These algorithm implementations are designed to mimic the characteristics of the
implementations provided by the original LensKit Java package.  It also provides abstract base
classes (:py:mod:`abc`) representing different algorithm capabilities.
"""

from abc import ABCMeta, abstractmethod
import inspect

__all__ = ['Algorithm', 'Recommender', 'Predictor', 'CandidateSelector']


class Algorithm(metaclass=ABCMeta):
    """
    Base class for LensKit algorithms.  These algorithms follow the SciKit design pattern
    for estimators.

    :canonical: lenskit.Algorithm
    """

    IGNORED_PARAMS = []
    """
    Names of parameters to ignore in :meth:`get_params`.
    """

    EXTRA_PARAMS = []
    """
    Names of extra parameters to include in :meth:`get_params`.  Useful when the
    constructor takes ``**kwargs``.
    """

    @abstractmethod
    def fit(self, ratings, **kwargs):
        """
        Train a model using the specified ratings (or similar) data.

        Args:
            ratings(pandas.DataFrame): The ratings data.
            kwargs: Additional training data the algorithm may require.  Algorithms should
                avoid using the same keyword arguments for different purposes, so that
                they can be more easily hybridized.

        Returns:
            The algorithm object.
        """
        raise NotImplementedError()

    def get_params(self, deep=True):
        """
        Get the parameters for this algorithm (as in scikit-learn).  Algorithm parameters
        should match constructor argument names.

        The default implementation returns all attributes that match a constructor parameter
        name.  It should be compatible with :py:meth:`sklearn.base.BaseEstimator.get_params`
        method so that LensKit alogrithms can be cloned with :py:func:`sklearn.base.clone`
        as well as :py:func:`lenskit.util.clone`.

        Returns:
            dict: the algorithm parameters.
        """
        sig = inspect.signature(self.__class__)
        names = list(sig.parameters.keys()) + self.EXTRA_PARAMS
        params = {}
        for name in names:
            if hasattr(self, name) and name not in self.IGNORED_PARAMS:
                value = getattr(self, name)
                params[name] = value
                if deep and hasattr(value, 'get_params'):
                    sps = value.get_params(deep)
                    for k, sv in sps.items():
                        params[name + '__' + k] = sv

        return params


class Predictor(Algorithm, metaclass=ABCMeta):
    """
    Predicts user ratings of items.  Predictions are really estimates of the user's like or
    dislike, and the ``Predictor`` interface makes no guarantees about their scale or
    granularity.

    :canonical: lenskit.Predictor
    """

    def predict(self, pairs, ratings=None):
        """
        Compute predictions for user-item pairs.  This method is designed to be compatible with the
        general SciKit paradigm; applications typically want to use :py:meth:`predict_for_user`.

        Args:
            pairs(pandas.DataFrame): The user-item pairs, as ``user`` and ``item`` columns.
            ratings(pandas.DataFrame): user-item rating data to replace memorized data.

        Returns:
            pandas.Series: The predicted scores for each user-item pair.
        """
        if ratings is not None:
            raise NotImplementedError()

        def upred(df):
            user, = df['user'].unique()
            items = df['item']
            preds = self.predict_for_user(user, items)
            preds.name = 'prediction'
            res = df.join(preds, on='item', how='left')
            return res.prediction

        res = pairs.loc[:, ['user', 'item']].groupby('user', sort=False).apply(upred)
        res.reset_index(level='user', inplace=True, drop=True)
        res.name = 'prediction'
        return res.loc[pairs.index.values]

    @abstractmethod
    def predict_for_user(self, user, items, ratings=None):
        """
        Compute predictions for a user and items.

        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series):
                the user's ratings (indexed by item id); if provided, they may be used to
                override or augment the model's notion of a user's preferences.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """
        raise NotImplementedError()


class Recommender(Algorithm, metaclass=ABCMeta):
    """
    Recommends lists of items for users.
    """

    @abstractmethod
    def recommend(self, user, n=None, candidates=None, ratings=None):
        """
        Compute recommendations for a user.

        Args:
            user: the user ID
            n(int): the number of recommendations to produce (``None`` for unlimited)
            candidates (array-like):
                The set of valid candidate items; if ``None``, a default set will be used.
                For many algorithms, this is their :py:class:`CandidateSelector`.
            ratings (pandas.Series):
                the user's ratings (indexed by item id); if provided, they may be used to
                override or augment the model's notion of a user's preferences.

        Returns:
            pandas.DataFrame:
                a frame with an ``item`` column; if the recommender also produces scores,
                they will be in a ``score`` column.
        """
        raise NotImplementedError()

    @classmethod
    def adapt(cls, algo):
        """
        Ensure that an algorithm is a :class:`Recommender`.  If it is not a recommender,
        it is wrapped in a :class:`lenskit.basic.TopN` with a default candidate selector.

        .. note::
            Since 0.6.0, since algorithms are fit directly, you should call this method
            **before** calling :meth:`Algorithm.fit`, unless you will always be passing
            explicit candidate sets to :meth:`recommend`.

        Args:
            algo(Predictor): the underlying rating predictor.
        """
        from .basic import TopN
        if isinstance(algo, Recommender):
            return algo
        else:
            return TopN(algo)


class CandidateSelector(Algorithm, metaclass=ABCMeta):
    """
    Select candidates for recommendation for a user, possibly with some
    additional ratings.

    :class:`.UnratedItemCandidateSelector` is the default and most common implementation
    of this interface.
    """

    @abstractmethod
    def candidates(self, user, ratings=None):
        """
        Select candidates for the user.

        Args:
            user:
                The user key or ID.
            ratings(pandas.Series or array-like):
                Ratings or items to use instead of whatever ratings were memorized
                for this user.  If a :py:class:`pandas.Series`, the series index
                is used; if it is another array-like it is assumed to be an array
                of items.
        """
        raise NotImplementedError()

    @staticmethod
    def rated_items(ratings):
        """
        Utility function for converting a series or array into an array of item
        IDs.  Useful in implementations of :py:meth:`candidates`.
        """
        import pandas as pd
        import numpy as np
        if isinstance(ratings, pd.Series):
            return ratings.index.values
        elif isinstance(ratings, np.ndarray):
            return ratings
        else:
            return np.array(ratings)
