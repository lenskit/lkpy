# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import hpfrec
from typing_extensions import override

from lenskit.algorithms.mf_common import MFPredictor
from lenskit.data import Dataset

_logger = logging.getLogger(__name__)


class HPF(MFPredictor):
    """
    Hierarchical Poisson factorization, provided by
    `hpfrec <https://hpfrec.readthedocs.io/en/latest/>`_.

    .. todo::
        Right now, this uses the 'rating' as a count. Actually use counts.

    Args:
        features(int): the number of features
        **kwargs: arguments passed to :py:class:`hpfrec.HPF`.
    """

    def __init__(self, features, **kwargs):
        self.features = features
        self._kwargs = kwargs

    @override
    def fit(self, data: Dataset, **kwargs):
        log = data.interaction_matrix("pandas", field="rating")
        log = log.rename(
            columns={
                "user_num": "UserId",
                "item_num": "ItemId",
                "rating": "Count",
            }
        )

        hpf = hpfrec.HPF(self.features, reindex=False, **self._kwargs)

        _logger.info("fitting HPF model with %d features", self.features)
        hpf.fit(log)

        self.users_ = data.users
        self.items_ = data.items
        self.user_features_ = hpf.Theta
        self.item_features_ = hpf.Beta

        return self

    def predict_for_user(self, user, items, ratings=None):
        # look up user index
        return self.score_by_ids(user, items)
