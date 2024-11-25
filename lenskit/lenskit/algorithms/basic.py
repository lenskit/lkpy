# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Basic utility algorithms and combiners.
"""

import logging
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import overload

import numpy as np
import pandas as pd
from typing_extensions import override

from lenskit.algorithms import CandidateSelector, Predictor, Recommender
from lenskit.algorithms.bias import Bias  # noqa: F401
from lenskit.algorithms.ranking import TopN  # noqa: F401
from lenskit.data import Dataset
from lenskit.data.matrix import CSRStructure
from lenskit.data.vocab import Vocabulary
from lenskit.util import derivable_rng

_logger = logging.getLogger(__name__)


class PopScore(Predictor):
    """
    Score items by their popularity.  Use with :py:class:`TopN` to get a
    most-popular-items recommender.

    Args:
        score_type(str):
            The method for computing popularity scores.  Can be one of the following:

            - ``'quantile'`` (the default)
            - ``'rank'``
            - ``'count'``

    Attributes:
        item_scores_(pandas.Series):
            Item popularity scores.
    """

    def __init__(self, score_method="quantile"):
        self.score_method = score_method

    @override
    def fit(self, data: Dataset, **kwargs):
        _logger.info("counting item popularity")

        counts = data.item_stats()["count"]
        self.item_scores_ = self._fit_internal(counts, **kwargs)

        return self

    def _fit_internal(self, scores: pd.Series, **kwargs):
        if self.score_method == "rank":
            _logger.info("ranking %d items", len(scores))
            scores = scores.rank().sort_index()
        elif self.score_method == "quantile":
            _logger.info("computing quantiles for %d items", len(scores))
            cmass = scores.sort_values()
            cmass = cmass.cumsum()
            cdens = cmass / scores.sum()
            scores = cdens.sort_index()
        elif self.score_method == "count":
            _logger.info("scoring items with their rating counts")
            scores = scores.sort_index()
        else:
            raise ValueError("invalid scoring method " + repr(self.score_method))

        return scores

    @override
    def predict_for_user(self, user, items, ratings=None):
        return self.item_scores_.reindex(items)

    def __str__(self):
        return "PopScore({})".format(self.score_method)


class TimeBoundedPopScore(PopScore):
    """
    Score items by their time-bounded popularity, i.e., the popularity in the
    most recent `time_window` period.  Use with :py:class:`TopN` to get a
    most-popular-recent-items recommender.

    Args:
        time_window(datetime.timedelta):
            The time window for computing popularity scores.
        score_type(str):
            The method for computing popularity scores.  Can be one of the following:

            - ``'quantile'`` (the default)
            - ``'rank'``
            - ``'count'``

    Attributes:
        item_scores_(pandas.Series):
            Time-bounded item popularity scores.
    """

    def __init__(self, cutoff: datetime, score_method="quantile"):
        super().__init__(score_method)

        self.cutoff = cutoff
        self.score_method = score_method

    @override
    def fit(self, data: Dataset, **kwargs):
        _logger.info("counting time-bounded item popularity")

        log = data.interaction_log("numpy")

        item_scores = None
        if log.timestamps is None:
            _logger.warning("no timestamps in interaction log; falling back to PopScore")
            item_scores = super().fit(data, **kwargs).item_scores_
        else:
            counts = np.zeros(data.item_count, dtype=np.int32)
            start_timestamp = self.cutoff.timestamp()
            item_nums = log.item_nums[log.timestamps > start_timestamp]
            np.add.at(counts, item_nums, 1)

            item_scores = super()._fit_internal(pd.Series(counts, index=data.items.index), **kwargs)

        self.item_scores_ = item_scores

        return self

    @override
    def __str__(self):
        return "TimeBoundedPopScore({}, {})".format(self.cutoff, self.score_method)


class Memorized(Predictor):
    """
    The memorized algorithm memorizes socres provided at construction time
    (*not* training time).
    """

    scores: pd.DataFrame

    def __init__(self, scores: pd.DataFrame):
        """
        Args:
            scores: the scores to memorize.
        """

        self.scores = scores

    @override
    def fit(self, *args, **kwargs):
        return self

    @override
    def predict_for_user(self, user, items, ratings=None):
        uscores = self.scores[self.scores.user == user]
        urates = uscores.set_index("item").rating
        return urates.reindex(items)


class Fallback(Predictor):
    """
    The Fallback algorithm predicts with its first component, uses the second to fill in
    missing values, and so forth.
    """

    algorithms: list[Predictor]

    @overload
    def __init__(self, algorithms: Iterable[Predictor]): ...
    @overload
    def __init__(self, algorithms: Predictor, *others: Predictor): ...
    def __init__(self, algorithms: Predictor | Iterable[Predictor], *others):
        """
        Args:
            algorithms: a list of component algorithms.  Each one will be trained.
            others:
                additional algorithms, in which case ``algorithms`` is taken to be
                a single algorithm.
        """
        if isinstance(algorithms, Iterable) or isinstance(algorithms, Sequence):
            assert not others
            self.algorithms = list(algorithms)
        else:
            self.algorithms = [algorithms] + list(others)

    @override
    def fit(self, data: Dataset, **kwargs):
        for algo in self.algorithms:
            algo.fit(data, **kwargs)

        return self

    @override
    def predict_for_user(self, user, items, ratings=None):
        remaining = pd.Index(items)
        preds = None

        for algo in self.algorithms:
            _logger.debug("predicting for %d items for user %s", len(remaining), user)
            aps = algo.predict_for_user(user, remaining, ratings=ratings)
            aps = aps[aps.notna()]
            if preds is None:
                preds = aps
            else:
                preds = pd.concat([preds, aps])
            remaining = remaining.difference(preds.index)
            if len(remaining) == 0:
                break

        assert preds is not None
        return preds.reindex(items)

    def __str__(self):
        str_algos = [str(algo) for algo in self.algorithms]
        return "Fallback([{}])".format(", ".join(str_algos))


class EmptyCandidateSelector(CandidateSelector):
    """
    :class:`CandidateSelector` that never returns any candidates.
    """

    dtype_ = np.int64

    @override
    def fit(self, data: Dataset, **kwarsg):
        self.dtype_ = data.items.index.dtype

    @override
    def candidates(self, user, ratings=None):
        return np.array([], dtype=self.dtype_)  # type: ignore


class UnratedItemCandidateSelector(CandidateSelector):
    """
    :class:`CandidateSelector` that selects items a user has not rated as
    candidates.  When this selector is fit, it memorizes the rated items.

    Attributes:
        items_: All known items.
        users_: All known users.
        user_items_:
            Items rated by each known user, as positions in the ``items`` index.
    """

    items_: Vocabulary
    users_: Vocabulary
    user_items_: CSRStructure

    @override
    def fit(self, data: Dataset, **kwargs):
        sparse = data.interaction_matrix(format="structure")
        _logger.info("trained unrated candidate selector for %d ratings", sparse.nnz)
        self.items_ = data.items.copy()
        self.users_ = data.users.copy()
        self.user_items_ = sparse

        return self

    def candidates(self, user, ratings=None):
        if ratings is None:
            try:
                uno = self.users_.number(user)
                uis = self.user_items_.row_cs(uno)
            except KeyError:
                uis = None
        else:
            uis = self.items_.numbers(self.rated_items(ratings))
            uis = uis[uis >= 0]

        if uis is not None:
            mask = np.full(len(self.items_), True)
            mask[uis] = False
            return self.items_.index.values[mask]
        else:
            return self.items_.index.values


class AllItemsCandidateSelector(CandidateSelector):
    """
    :class:`CandidateSelector` that selects all items, regardless of whether
    the user has rated them, as candidates.  When this selector is fit, it memorizes
    the set of items.

    Attributes:
        items_(numpy.ndarray): All known items.
    """

    items_: np.ndarray

    def fit(self, ratings, **kwargs):
        self.items_ = ratings["item"].unique()
        return self

    def candidates(self, user, ratings=None):
        return self.items_.copy()


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
        self.rng_source = derivable_rng(rng_spec)
        self.items_ = None

    @override
    def fit(self, data: Dataset, **kwargs):
        self.selector.fit(data, **kwargs)
        return self

    @override
    def recommend(self, user, n=None, candidates=None, ratings=None):
        if candidates is None:
            candidates = self.selector.candidates(user, ratings)
        if n is None:
            n = len(candidates)

        rng = self.rng_source(user)
        c_df = pd.DataFrame(candidates, columns=["item"])
        recs = c_df.sample(n, random_state=rng)
        return recs.reset_index(drop=True)

    def __str__(self):
        return "Random"


class KnownRating(Predictor):
    """
    The known rating algorithm memorizes ratings provided in the fit method.
    """

    ratings_: pd.DataFrame

    def fit(self, data: Dataset, **kwargs):
        self.ratings_ = (
            data.interaction_matrix(format="pandas", field="rating", original_ids=True)
            .set_index(["user_id", "item_id"])
            .sort_index()
        )
        return self

    def predict_for_user(self, user, items, ratings=None):
        uscores = self.ratings_.xs(user, level="user_id", drop_level=True)
        return uscores.rating.reindex(items)
