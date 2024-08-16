# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Algorithms to rank items based on scores.
"""

import logging

import numpy as np
from typing_extensions import override

from lenskit.data import Dataset

from ..util import derivable_rng
from . import Predictor, Recommender

_log = logging.getLogger(__name__)


class TopN(Recommender, Predictor):
    """
    Basic recommender that implements top-N recommendation using a predictor.

    .. note::
        This class does not do anything of its own in :meth:`fit`.  If its
        predictor and candidate selector are both fit separately, the top-N recommender
        does not need to be fit.  This can be useful when reusing a predictor in other
        contexts::

            pred = knn.ItemItem(20, feedback='implicit')
            select = UnratedItemCandidateSelector()
            topn = TopN(pred, select)

            pred.fit(ratings)
            select.fit(ratings)
            # topn.fit is unnecessary now

    Args:
        predictor(Predictor):
            The underlying predictor.
        selector(CandidateSelector):
            The candidate selector.  If ``None``, uses
            :class:`UnratedItemCandidateSelector`.
    """

    def __init__(self, predictor, selector=None):
        from .basic import UnratedItemCandidateSelector

        self.predictor = predictor
        self.selector = selector if selector is not None else UnratedItemCandidateSelector()

    @override
    def fit(self, data: Dataset, **kwargs):
        """
        Fit the recommender.

        Args:
            ratings:
                The rating or interaction data.  Passed changed to the predictor and
                candidate selector.
            args, kwargs:
                Additional arguments for the predictor to use in its training process.
        """
        self.predictor.fit(data, **kwargs)
        self.selector.fit(data, **kwargs)
        return self

    def fit_iters(self, ratings, **kwargs):
        if not hasattr(self.predictor, "fit_iters"):
            raise AttributeError("predictor has no method fit_iters")

        self.selector.fit(ratings, **kwargs)
        pred = self.predictor
        for p in pred.fit_iters(ratings, **kwargs):
            self.predictor = p
            yield self

        self.predictor = pred

    @override
    def recommend(self, user, n=None, candidates=None, ratings=None):
        if candidates is None:
            candidates = self.selector.candidates(user, ratings)

        scores = self.predictor.predict_for_user(user, candidates, ratings)
        scores = scores[scores.notna()]
        if n is not None:
            scores = scores.nlargest(n)
        else:
            scores = scores.sort_values(ascending=False)
        scores.name = "score"
        scores.index.name = "item"
        return scores.reset_index()

    @override
    def predict(self, pairs, ratings=None):
        return self.predictor.predict(pairs, ratings)

    @override
    def predict_for_user(self, user, items, ratings=None):
        return self.predictor.predict_for_user(user, items, ratings)

    def __str__(self):
        return "TopN/" + str(self.predictor)


class PlackettLuce(Recommender):
    """
    Re-ranking algorithm that uses Plackett-Luce sampling on underlying scores.
    This uses the Gumbel trick :cite:p:`groverStochasticOptimizationSorting2019`
    to efficiently simulate from a Plackett-Luce distribution.

    Args:
        predictor(Predictor):
            A predictor that can score candidate items.
        selector(CandidateSelector):
            The candidate selector. If ``None``, defaults to
            :py:class:`UnratedItemsCandidateSelector`.
        rng_spec:
            A random number generator specification; see
            :py:func:`derivable_rng`.
    """

    def __init__(self, predictor, selector=None, *, rng_spec=None):
        from .basic import UnratedItemCandidateSelector

        if isinstance(predictor, TopN):
            _log.warn("wrapping Top-N in PlackettLuce, candidate selector probably redundant")

        self.predictor = predictor
        self.selector = selector if selector is not None else UnratedItemCandidateSelector()
        self.rng_spec = rng_spec

    @override
    def fit(self, data: Dataset, **kwargs):
        self.predictor.fit(data, **kwargs)
        self.selector.fit(data, **kwargs)
        self.rng_ = derivable_rng(self.rng_spec)
        return self

    @override
    def recommend(self, user, n=None, candidates=None, ratings=None):
        if candidates is None:
            candidates = self.selector.candidates(user, ratings)

        rng = self.rng_(user)

        scores = self.predictor.predict_for_user(user, candidates)
        scores = scores.dropna()
        adjs = rng.gumbel(size=len(scores))
        scores = np.log(scores) + adjs

        if n is not None:
            scores = scores.nlargest(n)
        else:
            scores = scores.sort_values(ascending=False)
        scores.name = "score"
        scores.index.name = "item"
        return scores.reset_index()
