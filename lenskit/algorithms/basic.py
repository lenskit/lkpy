"""
Basic utility algorithms and combiners.
"""

from collections import namedtuple
import logging
import pathlib

import pandas as pd

from .. import util as lku
from .. import check
from . import Predictor, Trainable, Recommender

_logger = logging.getLogger(__package__)

BiasModel = namedtuple('BiasModel', ['mean', 'items', 'users'])
BiasModel.__doc__ = "Trained model for the :py:class:`Bias` algorithm."


class Bias(Predictor, Trainable):
    """
    A user-item bias rating prediction algorithm.  This implements the following
    predictor algorithm:

    .. math::
       s(u,i) = \mu + b_i + b_u

    where :math:`\mu` is the global mean rating, :math:`b_i` is item bias, and
    :math:`b_u` is the user bias.  With the provided damping values
    :math:`\\beta_{\\mathrm{u}}` and :math:`\\beta_{\\mathrm{i}}`, they are computed
    as follows:

    .. math::
       \\begin{align*}
       \\mu & = \\frac{\\sum_{r_{ui} \\in R} r_{ui}}{|R|} &
       b_i & = \\frac{\\sum_{r_{ui} \\in R_i} (r_{ui} - \\mu)}{|R_i| + \\beta_{\\mathrm{i}}} &
       b_u & = \\frac{\\sum_{r_{ui} \\in R_u} (r_{ui} - \\mu - b_i)}{|R_u| + \\beta_{\\mathrm{u}}}
       \\end{align*}

    The damping values can be interpreted as the number of default (mean) ratings to assume
    *a priori* for each user or item, damping low-information users and items towards a mean instead
    of permitting them to take on extreme values based on few ratings.

    Args:
        items: whether to compute item biases
        users: whether to compute user biases
        damping(number or tuple):
            Bayesian damping to apply to computed biases.  Either a number, to
            damp both user and item biases the same amount, or a (user,item) tuple
            providing separate damping values.
    """

    def __init__(self, items=True, users=True, damping=0.0):
        self._include_items = items
        self._include_users = users
        if isinstance(damping, tuple):
            self.user_damping, self.item_damping = damping
        else:
            self.user_damping = damping
            self.item_damping = damping

        check.check_value(self.user_damping >= 0, "user damping value {} must be nonnegative",
                          self.user_damping)
        check.check_value(self.item_damping >= 0, "item damping value {} must be nonnegative",
                          self.item_damping)

    def train(self, data):
        """
        Train the bias model on some rating data.

        Args:
            data (DataFrame): a data frame of ratings. Must have at least `user`,
                              `item`, and `rating` columns.

        Returns:
            BiasModel: a trained model with the desired biases computed.
        """

        _logger.info('building bias model for %d ratings', len(data))
        mean = data.rating.mean()
        _logger.info('global mean: %.3f', mean)
        nrates = data.assign(rating=lambda df: df.rating - mean)

        if self._include_items:
            group = nrates.groupby('item').rating
            item_offsets = self._mean(group, self.item_damping)
            _logger.info('computed means for %d items', len(item_offsets))
        else:
            item_offsets = None

        if self._include_users:
            if item_offsets is not None:
                nrates = nrates.join(pd.DataFrame(item_offsets), on='item', how='inner',
                                     rsuffix='_im')
                nrates = nrates.assign(rating=lambda df: df.rating - df.rating_im)

            user_offsets = self._mean(nrates.groupby('user').rating, self.user_damping)
            _logger.info('computed means for %d users', len(user_offsets))
        else:
            user_offsets = None

        return BiasModel(mean, item_offsets, user_offsets)

    def predict(self, model, user, items, ratings=None):
        """
        Compute predictions for a user and items.  Unknown users and items
        are assumed to have zero bias.

        Args:
            model (BiasModel): the trained model to use.
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series): the user's ratings (indexed by item id); if
                                 provided, will be used to recompute the user's
                                 bias at prediction time.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """

        idx = pd.Index(items)
        preds = pd.Series(model.mean, idx)

        if model.items is not None:
            preds = preds + model.items.reindex(items, fill_value=0)

        if self._include_users and ratings is not None:
            uoff = ratings - model.mean
            if model.items is not None:
                uoff = uoff - model.items
            umean = uoff.mean()
            preds = preds + umean
        elif model.users is not None:
            umean = model.users.get(user, 0.0)
            _logger.debug('using mean(user %s) = %.3f', user, umean)
            preds = preds + umean

        return preds

    def _mean(self, series, damping):
        if damping is not None and damping > 0:
            return series.sum() / (series.count() + damping)
        else:
            return series.mean()


class Popular(Recommender, Trainable):
    def train(self, ratings):
        pop = ratings.groupby('item').user.count()
        pop.name = 'score'
        return pop

    def recommend(self, model, user, n=None, candidates=None, ratings=None):
        scores = model
        if candidates is not None:
            idx = scores.index.get_indexer(candidates)
            idx = idx[idx >= 0]
            scores = scores.iloc[idx]

        if n is None:
            return scores.sort_values(ascending=False).reset_index()
        else:
            return scores.nlargest(n).reset_index()


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

    def predict(self, model, user, items, ratings=None):
        uscores = self.scores[self.scores.user == user]
        urates = uscores.set_index('item').rating
        return urates.reindex(items)


class Fallback(Predictor, Trainable):
    """
    The Fallback algorithm predicts with its first component, uses the second to fill in
    missing values, and so forth.
    """

    def __init__(self, *algorithms):
        """
        Args:
            algorithms: a list of component algorithms.  Each one will be trained.
        """
        self.algorithms = algorithms

    def train(self, ratings):
        models = []
        for a in self.algorithms:
            if isinstance(a, Trainable):
                models.append(a.train(ratings))
            else:
                models.append(None)

        return models

    def predict(self, model, user, items, ratings=None):
        remaining = pd.Index(items)
        preds = None

        for algo, amod in zip(self.algorithms, model):
            _logger.debug('predicting for %d items for user %s', len(remaining), user)
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

    def save_model(self, model, file):
        path = pathlib.Path(file)
        path.mkdir(parents=True, exist_ok=True)
        for i, algo in enumerate(self.algorithms):
            mp = path / 'algo-{}.dat'.format(i+1)
            mod = model[i]
            if mod is not None:
                _logger.debug('saving {} to {}', mod, mp)
                algo.save_model(mod, str(mp))

    def load_model(self, file):
        path = pathlib.Path(file)

        model = []

        for i, algo in enumerate(self.algorithms):
            mp = path / 'algo-{}.dat'.format(i+1)
            if mp.exists():
                _logger.debug('loading {} from {}', algo, mp)
                model.append(algo.load_model(str(mp)))
            else:
                model.append(None)

        return model


class TopN(Recommender):
    """
    Basic recommender that implements top-N recommendation using a predictor.

    Args:
        predictor(Predictor):
            the underlying predictor.  If it is :py:class:`Trainable`, then the resulting
            recommender class will also be :py:class:`Trainable`.
    """

    def __new__(cls, predictor):
        if isinstance(predictor, Trainable):
            return super().__new__(_TrainableTopN)
        else:
            return super().__new__(cls)

    def __init__(self, predictor):
        self.predictor = predictor

    def recommend(self, model, user, n=None, candidates=None, ratings=None):
        scores = self.predictor.predict(model, user, candidates, ratings)
        scores = scores[scores.notna()]
        scores = scores.sort_values(ascending=False)
        if n is not None:
            scores = scores.iloc[:n]
        scores.name = 'score'
        scores.index.name = 'item'
        return scores.reset_index()


class _TrainableTopN(TopN, Trainable):
    """
    Trainable subclass of :py:class:`TopN`.
    """

    def train(self, ratings):
        return self.predictor.train(ratings)

    def save_model(self, model, file):
        self.predictor.save_model(model, file)

    def load_model(self, file):
        return self.predictor.load_model(file)
