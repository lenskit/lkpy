# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
"""
Bias scoring model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Self, TypeAlias

import numpy as np

from lenskit.data import ID, Dataset, ItemList, QueryInput, RecQuery, UITuple, Vocabulary
from lenskit.pipeline import Component

_logger = logging.getLogger(__name__)
Damping: TypeAlias = float | UITuple[float] | tuple[float, float]


@dataclass
class BiasModel:
    """
    User-item bias models learned from rating data.  The :class:`BiasScorer`
    class uses this model to score items in a pipeline; the model is reusable
    in other components that need user-item bias models.

    This implements the following model:

    .. math::
       b_{ui} = b_g + b_i + b_u

    where :math:`b_g` is the global bias (global mean rating), :math:`b_i` is
    item bias, and :math:`b_u` is the user bias.  With the provided damping
    values :math:`\\beta_{\\mathrm{u}}` and :math:`\\beta_{\\mathrm{i}}`, they
    are computed as follows:

    .. math::
       \\begin{align*}
       b_g & = \\frac{\\sum_{r_{ui} \\in R} r_{ui}}{|R|} &
       b_i & = \\frac{\\sum_{r_{ui} \\in R_i} (r_{ui} - b_g)}{|R_i| + \\beta_{\\mathrm{i}}} &
       b_u & = \\frac{\\sum_{r_{ui} \\in R_u} (r_{ui} - b_g - b_i)}{|R_u| + \\beta_{\\mathrm{u}}}
       \\end{align*}

    The damping values can be interpreted as the number of default (mean)
    ratings to assume *a priori* for each user or item, damping low-information
    users and items towards a mean instead of permitting them to take on extreme
    values based on few ratings.
    """

    global_bias: float
    "The global bias term."

    items: Vocabulary | None = None
    "Vocabulary of items."
    item_biases: np.ndarray[int, np.dtype[np.float32]] | None = None
    "The item offsets (:math:`b_i` values)."

    users: Vocabulary | None = None
    "Vocabulary of users."
    user_biases: np.ndarray[int, np.dtype[np.float32]] | None = None
    "The user offsets (:math:`b_u` values)."

    @classmethod
    def learn(
        cls, data: Dataset, damping: Damping = 0.0, *, items: bool = True, users: bool = True
    ) -> Self:
        """
        Learn a bias model and its parameters from a dataset.

        Args:
            data:
                The dataset from which to learn the bias model.
            damping:
                Bayesian damping to apply to computed biases.  Either a number, to
                damp both user and item biases the same amount, or a (user,item)
                tuple providing separate damping values.
            items:
                Whether to compute item biases
            users:
                Whether to compute user biases
        """
        damping = UITuple.create(damping)

        _logger.info("building bias model for %d ratings", data.interaction_count)
        ratings = data.interaction_matrix("scipy", layout="coo", field="rating")
        nrows, ncols = ratings.shape  # type: ignore

        g_bias = float(np.mean(ratings.data))
        _logger.info("global mean: %.3f", g_bias)

        model = cls(g_bias)

        centered = ratings.data - g_bias
        if np.allclose(centered, 0):
            _logger.warning("mean-centered ratings are all 0, bias probably meaningless")

        if items:
            counts = np.full(ncols, damping.item)
            sums = np.zeros(ncols)
            np.add.at(counts, ratings.col, 1)
            np.add.at(sums, ratings.col, centered)

            # store 0 offsets
            i_bias = np.zeros(ncols, dtype=np.float32)
            np.divide(sums, counts, out=i_bias, where=counts > 0)

            model.items = data.items.copy()
            model.item_biases = i_bias
            centered -= i_bias[ratings.col]
            _logger.info("computed biases for %d items", len(i_bias))

        if users:
            counts = np.full(nrows, damping.user)
            sums = np.zeros(nrows)
            np.add.at(counts, ratings.row, 1)
            np.add.at(sums, ratings.row, centered)

            u_bias = np.zeros(nrows, dtype=np.float32)
            np.divide(sums, counts, out=u_bias, where=counts > 0)

            model.users = data.users.copy()
            model.user_biases = u_bias
            _logger.info("computed biases for %d users", len(u_bias))

        return model

    def compute_for_items(
        self, items: ItemList, user_id: ID | None = None, user_items: ItemList | None = None
    ) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        """
        Compute the personalized biases for a set of itemsm and optionally a
        user.  The user can be specified either by their identifier or by a list
        of ratings.

        Args:
            items:
                The items to score.
            user:
                The user identifier.
            user_items:
                The user's items, with ratings (takes precedence over ``user``
                if both are supplied).  If the supplied list does not have a
                ``rating`` field, it is ignored.
        """
        n = len(items)
        scores = np.full(n, self.global_bias, dtype=np.float32)

        if self.item_biases is not None:
            assert self.items is not None
            idxes = items.numbers(vocabulary=self.items, missing="negative")
            mask = idxes >= 0
            scores[mask] += self.item_biases[idxes[mask]]

        ratings = None
        if user_items is not None:
            ratings = user_items.field("rating")

        if self.users is not None:
            assert self.user_biases is not None

            if ratings is not None:
                assert user_items is not None

                uoff = ratings - self.global_bias

                if self.item_biases is not None:
                    r_idxes = user_items.numbers(vocabulary=self.items, missing="negative")
                    r_mask = r_idxes >= 0
                    uoff[r_mask] -= self.item_biases[r_idxes[r_mask]]

                umean = uoff.mean()
                scores += umean

            elif user_id is not None:
                uno = self.users.number(user_id, missing="none")
                if uno is not None:
                    umean = self.user_biases[uno]
                    _logger.debug("using mean(user %s) = %.3f", user_id, umean)
                    scores += umean

        return scores


class BiasScorer(Component):
    """
    A user-item bias rating prediction model.  This component uses
    :class:`BiasModel` to predict ratings for users and items.

    Args:
        items:
            Whether to compute item biases.
        users:
            Whether to compute user biases.
        damping:
            Bayesian damping to apply to computed biases.  Either a number, to
            damp both user and item biases the same amount, or a (user,item)
            tuple providing separate damping values.
    """

    users: bool
    items: bool
    damping: UITuple[float]
    "The configured offset damping levels."

    model_: BiasModel

    def __init__(
        self,
        items: bool = True,
        users: bool = True,
        damping: float | UITuple[float] | tuple[float, float] = 0.0,
    ):
        self.items = items
        self.users = users
        self.damping = UITuple.create(damping)

        if self.damping.user < 0:
            raise ValueError("user damping must be non-negative")
        if self.damping.item < 0:
            raise ValueError("item damping must be non-negative")

    def train(self, data: Dataset):
        """
        Train the bias model on some rating data.

        Args:
            ratings:
                The training data (must have ratings).

        Returns:
            The trained bias object.
        """
        self.model_ = BiasModel.learn(data, self.damping, users=self.users, items=self.items)

    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        """
        Compute predictions for a user and items.  Unknown users and items are
        assumed to have zero bias.

        Args:
            query:
                The recommendation query.  If the query has an item list with
                ratings, those ratings are used to compute a new bias instead of
                using the user's recorded bias.
            items:
                The items to score.
        Returns:
            Scores for `items`.
        """
        query = RecQuery.create(query)

        scores = self.model_.compute_for_items(items, query.user_id, query.user_items)
        return ItemList(items, scores=scores)

    def __str__(self):
        return "Bias(ud={}, id={})".format(self.damping.user, self.damping.item)
