"""
Basic utility algorithms and combiners.
"""

import logging
import pathlib
from collections.abc import Iterable, Sequence

import pandas as pd

from .. import check
from . import Predictor, Recommender

_logger = logging.getLogger(__name__)


class Bias(Predictor):
    """
    A user-item bias rating prediction algorithm.  This implements the following
    predictor algorithm:

    .. math::
       s(u,i) = \\mu + b_i + b_u

    where :math:`\\mu` is the global mean rating, :math:`b_i` is item bias, and
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

    Attributes:
        mean_(double): The global mean rating.
        item_offsets_(pandas.Series): The item offsets (:math:`b_i` values)
        user_offsets_(pandas.Series): The item offsets (:math:`b_u` values)
    """

    def __init__(self, items=True, users=True, damping=0.0):
        self.items = items
        self.users = users
        if isinstance(damping, tuple):
            self.damping = damping
            self.user_damping, self.item_damping = damping
        else:
            self.damping = damping
            self.user_damping = damping
            self.item_damping = damping

        check.check_value(self.user_damping >= 0, "user damping value {} must be nonnegative",
                          self.user_damping)
        check.check_value(self.item_damping >= 0, "item damping value {} must be nonnegative",
                          self.item_damping)

    def fit(self, data):
        """
        Train the bias model on some rating data.

        Args:
            data (DataFrame): a data frame of ratings. Must have at least `user`,
                              `item`, and `rating` columns.

        Returns:
            Bias: the fit bias object.
        """

        _logger.info('building bias model for %d ratings', len(data))
        self.mean_ = data.rating.mean()
        _logger.info('global mean: %.3f', self.mean_)
        nrates = data.assign(rating=lambda df: df.rating - self.mean_)

        if self.items:
            group = nrates.groupby('item').rating
            self.item_offsets_ = self._mean(group, self.item_damping)
            _logger.info('computed means for %d items', len(self.item_offsets_))
        else:
            self.item_offsets_ = None

        if self.users:
            if self.item_offsets_ is not None:
                nrates = nrates.join(pd.DataFrame(self.item_offsets_), on='item', how='inner',
                                     rsuffix='_im')
                nrates = nrates.assign(rating=lambda df: df.rating - df.rating_im)

            self.user_offsets_ = self._mean(nrates.groupby('user').rating, self.user_damping)
            _logger.info('computed means for %d users', len(self.user_offsets_))
        else:
            self.user_offsets_ = None

        return self

    def predict_for_user(self, user, items, ratings=None):
        """
        Compute predictions for a user and items.  Unknown users and items
        are assumed to have zero bias.

        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series): the user's ratings (indexed by item id); if
                                 provided, will be used to recompute the user's
                                 bias at prediction time.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """

        idx = pd.Index(items)
        preds = pd.Series(self.mean_, idx)

        if self.item_offsets_ is not None:
            preds = preds + self.item_offsets_.reindex(items, fill_value=0)

        if self.users and ratings is not None:
            uoff = ratings - self.mean_
            if self.item_offsets_ is not None:
                uoff = uoff - self.item_offsets_
            umean = uoff.mean()
            preds = preds + umean
        elif self.user_offsets_ is not None:
            umean = self.user_offsets_.get(user, 0.0)
            _logger.debug('using mean(user %s) = %.3f', user, umean)
            preds = preds + umean

        return preds

    def _mean(self, series, damping):
        if damping is not None and damping > 0:
            return series.sum() / (series.count() + damping)
        else:
            return series.mean()

    def __str__(self):
        return 'Bias(ud={}, id={})'.format(self.user_damping, self.item_damping)


class Popular(Recommender):
    def fit(self, ratings):
        pop = ratings.groupby('item').user.count()
        pop.name = 'score'
        self.item_pop_ = pop

        return self

    def recommend(self, user, n=None, candidates=None, ratings=None):
        scores = self.item_pop_
        if candidates is not None:
            idx = scores.index.get_indexer(candidates)
            idx = idx[idx >= 0]
            scores = scores.iloc[idx]

        if n is None:
            return scores.sort_values(ascending=False).reset_index()
        else:
            return scores.nlargest(n).reset_index()

    def __str__(self):
        return 'Popular'


class Memorized(Predictor):
    """
    The memorized algorithm memorizes socres provided at construction time.
    """

    def __init__(self, scores):
        """
        Args:
            scores(pandas.DataFrame): the scores to memorize.
        """

        self.scores = scores

    def fit(self, *args, **kwargs):
        return self

    def predict_for_user(self, user, items, ratings=None):
        uscores = self.scores[self.scores.user == user]
        urates = uscores.set_index('item').rating
        return urates.reindex(items)


class Fallback(Predictor):
    """
    The Fallback algorithm predicts with its first component, uses the second to fill in
    missing values, and so forth.
    """

    def __init__(self, algorithms, *others):
        """
        Args:
            algorithms: a list of component algorithms.  Each one will be trained.
            others:
                additional algorithms, in which case ``algorithms`` is taken to be
                a single algorithm.
        """
        if others:
            self.algorithms = [algorithms] + list(others)
        elif isinstance(algorithms, Iterable) or isinstance(algorithms, Sequence):
            self.algorithms = algorithms
        else:
            self.algorithms = [algorithms]

    def fit(self, ratings, *args, **kwargs):
        for algo in self.algorithms:
            algo.fit(ratings, *args, **kwargs)

        return self

    def predict_for_user(self, user, items, ratings=None):
        remaining = pd.Index(items)
        preds = None

        for algo in self.algorithms:
            _logger.debug('predicting for %d items for user %s', len(remaining), user)
            aps = algo.predict_for_user(user, remaining, ratings=ratings)
            aps = aps[aps.notna()]
            if preds is None:
                preds = aps
            else:
                preds = pd.concat([preds, aps])
            remaining = remaining.difference(preds.index)
            if len(remaining) == 0:
                break

        return preds.reindex(items)

    def save(self, path):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i, algo in enumerate(self.algorithms):
            mp = path / 'algo-{}.dat'.format(i+1)
            _logger.debug('saving {} to {}', algo, mp)
            algo.save(mp)

    def load(self, file):
        path = pathlib.Path(file)

        for i, algo in enumerate(self.algorithms):
            mp = path / 'algo-{}.dat'.format(i+1)
            _logger.debug('loading {} from {}', algo, mp)
            algo.load(mp)

    def __str__(self):
        return 'Fallback([{}])'.format(', '.join(self.algorithms))


class TopN(Recommender):
    """
    Basic recommender that implements top-N recommendation using a predictor.

    Args:
        predictor(Predictor):
            The underlying predictor.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def fit(self, ratings, *args, **kwargs):
        self.predictor.fit(ratings, *args, **kwargs)
        return self

    def recommend(self, user, n=None, candidates=None, ratings=None):
        scores = self.predictor.predict_for_user(user, candidates, ratings)
        scores = scores[scores.notna()]
        scores = scores.sort_values(ascending=False)
        if n is not None:
            scores = scores.iloc[:n]
        scores.name = 'score'
        scores.index.name = 'item'
        return scores.reset_index()

    def __str__(self):
        return 'TN/' + str(self.predidctor)


class Random(Recommender):
    """
    The Random algorithm recommends random items from all the items or candidate items(if provided).
    """

    def __init__(self, random_state=None):
        """
        Args:
             random_state: int or Series, optional
                Seed/Seeds for the random sampling of items. If int, then recommending random items
                for each user with the same seed. If Series, then it is the seeds for the users,
                indexed by user id.
        """
        self.random_state = random_state
        self.items = None

    def fit(self, ratings, *args, **kwargs):
        items = pd.DataFrame(ratings['item'].unique(), columns=['item'])
        self.items = items
        return self

    def recommend(self, user, n=None, candidates=None, ratings=None):
        model = self.items
        seed = None
        if isinstance(self.random_state, int):
            seed = self.random_state
        if isinstance(self.random_state, pd.Series):
            seed = self.random_state.get(user)

        frac = None
        if n is None:
            frac = 1

        if candidates is not None:
            return (pd.DataFrame(candidates, columns=['item'])
                    .sample(n, frac, random_state=seed)
                    .reset_index(drop=True))
        else:
            return (model.sample(n, frac, random_state=seed)
                    .reset_index(drop=True))

    def __str__(self):
        return 'Random'
