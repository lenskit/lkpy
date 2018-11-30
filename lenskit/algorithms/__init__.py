"""
LensKit algorithms.

The `lenskit.algorithms` package contains several example algorithms for carrying out recommender
experiments.  These algorithm implementations are designed to mimic the characteristics of the
implementations provided by the original LensKit Java package.  It also provides abstract base
classes (:py:mod:`abc`) representing different algorithm capabilities.
"""

from abc import ABCMeta, abstractmethod
import pickle
import pathlib


class Algorithm(metaclass=ABCMeta):
    """
    Base class for LensKit algorithms.  These algorithms follow the SciKit design pattern
    for estimators.
    """

    @abstractmethod
    def fit(self, ratings, *args, **kwargs):
        """
        Train a model using the specified ratings (or similar) data.

        Args:
            ratings(pandas.DataFrame): The ratings data.
            args: Additional training data the algorithm may require.
            kwargs: Additional training data the algorithm may require.

        Returns:
            The algorithm object.
        """
        raise NotImplementedError()

    def save(self, file):
        """
        Save a fit algorithm to a file.  The default implementation pickles the object.

        Args:
            file(path-like): the file to save.
        """
        path = pathlib.Path(file)
        with path.open('wb') as f:
            pickle.dump(self, f)

    def load(self, file):
        """
        Load a fit algorithm from a file.  The default implementation unpickles the object
        and transplants its parameters into this object.

        Args:
            file(path-like): the file to load.
        """
        path = pathlib.Path(file)
        with path.open('rb') as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)


class Predictor(Algorithm, metaclass=ABCMeta):
    """
    Predicts user ratings of items.  Predictions are really estimates of the user's like or
    dislike, and the ``Predictor`` interface makes no guarantees about their scale or
    granularity.
    """

    def skpredict(self, pairs, ratings=None):
        """
        Compute predictions for user-item pairs.  This method is designed to be compatible with the
        general SciKit paradigm; applications typically want to use :py:meth:`predict_for_user`.

        Args:
            pairs(pandas.DataFrame): The user-item pairs, as ``user`` and ``item`` columns.
            ratings(pandas.DataFrame): user-item rating data to replace memorized data.

        Returns:
            pandas.DataFrame: The predicted scores for the items, in a `prediction` column.
        """

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
            candidates (array-like): the set of valid candidate items.
            ratings (pandas.Series):
                the user's ratings (indexed by item id); if provided, they may be used to
                override or augment the model's notion of a user's preferences.

        Returns:
            pandas.DataFrame:
                a frame with an ``item`` column; if the recommender also produces scores,
                they will be in a ``score`` column.
        """
        raise NotImplementedError()
