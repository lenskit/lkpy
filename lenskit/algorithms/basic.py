"""
Basic utility algorithms and combiners.
"""

import logging
from collections.abc import Iterable, Sequence

import pandas as pd
import numpy as np

from ..matrix import sparse_ratings
from . import Predictor, Recommender, CandidateSelector
from ..util import derivable_rng

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

        if self.user_damping < 0:
            raise ValueError("user damping must be non-negative")
        if self.item_damping < 0:
            raise ValueError("item damping must be non-negative")

    def fit(self, ratings, **kwargs):
        """
        Train the bias model on some rating data.

        Args:
            ratings (DataFrame): a data frame of ratings. Must have at least `user`,
                                 `item`, and `rating` columns.

        Returns:
            Bias: the fit bias object.
        """

        _logger.info('building bias model for %d ratings', len(ratings))
        self.mean_ = ratings.rating.mean()
        _logger.info('global mean: %.3f', self.mean_)
        nrates = ratings.assign(rating=lambda df: df.rating - self.mean_)

        if self.items:
            group = nrates.groupby('item').rating
            self.item_offsets_ = self._mean(group, self.item_damping)
            self.item_offsets_.name = 'i_off'
            _logger.info('computed means for %d items', len(self.item_offsets_))
        else:
            self.item_offsets_ = None

        if self.users:
            if self.item_offsets_ is not None:
                nrates = nrates.join(pd.DataFrame(self.item_offsets_), on='item', how='inner')
                nrates = nrates.assign(rating=lambda df: df.rating - df.i_off)

            self.user_offsets_ = self._mean(nrates.groupby('user').rating, self.user_damping)
            self.user_offsets_.name = 'u_off'
            _logger.info('computed means for %d users', len(self.user_offsets_))
        else:
            self.user_offsets_ = None

        return self

    def transform(self, ratings, *, indexes=False):
        """
        Transform ratings by removing the bias term.  This method does *not* recompute
        user (or item) biases based on these ratings, but rather uses the biases that
        were estimated with :meth:`fit`.

        Args:
            ratings(pandas.DataFrame):
                The ratings to transform.  Must contain at least ``user``, ``item``, and
                ``rating`` columns.
            indexes(bool):
                if ``True``, the resulting frame will include ``uidx`` and ``iidx``
                columns containing the 0-based user and item indexes for each rating.

        Returns:
            pandas.DataFrame:
                A data frame with ``rating`` transformed by subtracting
                user-item bias prediction.
        """
        rvps = ratings[['user', 'item']].copy()
        rvps['rating'] = ratings['rating'] - self.mean_
        if self.item_offsets_ is not None:
            rvps = rvps.join(self.item_offsets_, on='item', how='left')
            rvps['rating'] -= rvps['i_off'].fillna(0)
        if self.user_offsets_ is not None:
            rvps = rvps.join(self.user_offsets_, on='item', how='left')
            rvps['rating'] -= rvps['u_off'].fillna(0)
        if indexes:
            rvps['uidx'] = self.user_offsets_.index.get_indexer(rvps['user'])
            rvps['iidx'] = self.item_offsets_.index.get_indexer(rvps['item'])
        return rvps.drop(columns=['u_off', 'i_off'])

    def inverse_transform(self, ratings):
        """
        Transform ratings by removing the bias term.
        """
        rvps = ratings[['user', 'item']]
        rvps['rating'] = ratings['rating'] + self.mean_
        if self.item_offsets_ is not None:
            rvps = rvps.join(self.item_offsets_, on='item', how='left')
            rvps['rating'] += rvps['i_off'].fillna(0)
        if self.user_offsets_ is not None:
            rvps = rvps.join(self.user_offsets_, on='item', how='left')
            rvps['rating'] += rvps['u_off'].fillna(0)
        return rvps.drop(columns=['u_off', 'i_off'])

    def fit_transform(self, ratings, **kwargs):
        """
        Fit with ratings and return the training data transformed.
        """
        self.fit(ratings)
        return self.transform(ratings, **kwargs)

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
            preds = preds + self.item_offsets_.reindex(idx, fill_value=0)

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

    @property
    def user_index(self):
        "Get the user index from this (fit) bias."
        return self.user_offsets_.index

    @property
    def item_index(self):
        "Get the item index from this (fit) bias."
        return self.item_offsets_.index

    def _mean(self, series, damping):
        if damping is not None and damping > 0:
            return series.sum() / (series.count() + damping)
        else:
            return series.mean()

    def __str__(self):
        return 'Bias(ud={}, id={})'.format(self.user_damping, self.item_damping)


class Popular(Recommender):
    """
    Recommend the most popular items.

    Args:
        selector(CandidateSelector):
            The candidate selector to use. If ``None``, uses a new
            :class:`UnratedItemCandidateSelector`.

    Attributes:
        item_pop_(pandas.Series):
            Item rating counts (popularity)
    """

    def __init__(self, selector=None):
        if selector is None:
            self.selector = UnratedItemCandidateSelector()
        else:
            self.selector = selector

    def fit(self, ratings, **kwargs):
        pop = ratings.groupby('item').user.count()
        pop.name = 'score'
        self.item_pop_ = pop.astype('float64')
        self.selector.fit(ratings)

        return self

    def recommend(self, user, n=None, candidates=None, ratings=None):
        scores = self.item_pop_
        if candidates is None:
            candidates = self.selector.candidates(user, ratings)

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

    def fit(self, ratings, **kwargs):
        for algo in self.algorithms:
            algo.fit(ratings, **kwargs)

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

    def __str__(self):
        str_algos = [str(algo) for algo in self.algorithms]
        return 'Fallback([{}])'.format(', '.join(str_algos))


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


class UnratedItemCandidateSelector(CandidateSelector):
    """
    :class:`CandidateSelector` that selects items a user has not rated as
    candidates.  When this selector is fit, it memorizes the rated items.

    Attributes:
        items_(pandas.Index): All known items.
        users_(pandas.Index): All known users.
        user_items_(CSR):
            Items rated by each known user, as positions in the ``items`` index.
    """
    items_ = None
    users_ = None
    user_items_ = None

    def fit(self, ratings, **kwargs):
        r2 = ratings[['user', 'item']]
        sparse = sparse_ratings(r2)
        _logger.info('trained unrated candidate selector for %d ratings', sparse.matrix.nnz)
        self.items_ = sparse.items
        self.users_ = sparse.users
        self.user_items_ = sparse.matrix

        return self

    def candidates(self, user, ratings=None):
        if ratings is None:
            try:
                uidx = self.users_.get_loc(user)
                uis = self.user_items_.row_cs(uidx)
            except KeyError:
                uis = None
        else:
            uis = self.items_.get_indexer(self.rated_items(ratings))
            uis = uis[uis >= 0]

        if uis is not None:
            mask = np.full(len(self.items_), True)
            mask[uis] = False
            return self.items_.values[mask]
        else:
            return self.items_.values


class Random(Recommender):
    """
    A random-item recommender.

    Attributes:
        selector(CandidateSelector):
            Selects candidate items for recommendation.
            Default is :class:`UnratedItemCandidateSelector`.
        rng_spec:
            Seed or random state for generating recommendations.  Pass
            ``'user'`` to deterministically derive per-user RNGS from
            the user IDs for reproducibility.
    """

    def __init__(self, selector=None, rng_spec=None):
        if selector:
            self.selector = selector
        else:
            self.selector = UnratedItemCandidateSelector()
        # Get a Pandas-compatible RNG
        self.rng_source = derivable_rng(rng_spec, legacy=True)
        self.items = None

    def fit(self, ratings, **kwargs):
        self.selector.fit(ratings, **kwargs)
        items = pd.DataFrame(ratings['item'].unique(), columns=['item'])
        self.items = items
        return self

    def recommend(self, user, n=None, candidates=None, ratings=None):
        if candidates is None:
            candidates = self.selector.candidates(user, ratings)
        if n is None:
            n = len(candidates)

        rng = self.rng_source(user)
        c_df = pd.DataFrame(candidates, columns=['item'])
        recs = c_df.sample(n, random_state=rng)
        return recs.reset_index(drop=True)

    def __str__(self):
        return 'Random'
