# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
FunkSVD (biased MF).
"""

import time
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from pydantic import AliasChoices, BaseModel, Field, NonNegativeFloat, PositiveFloat, PositiveInt
from typing_extensions import override

from lenskit._accel import FunkSVDTrainer
from lenskit.basic import BiasModel, Damping
from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.data.types import NPMatrix
from lenskit.logging import Stopwatch, get_logger
from lenskit.logging.progress._dispatch import item_progress
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_logger = get_logger(__name__)

INITIAL_VALUE = 0.1


class FunkSVDConfig(BaseModel):
    "Configuration for :class:`FunkSVDScorer`."

    embedding_size: PositiveInt = Field(
        default=50, validation_alias=AliasChoices("embedding_size", "features")
    )
    """
    Number of latent features.
    """
    epochs: PositiveInt = 100
    """
    Number of training epochs (per feature).
    """
    learning_rate: PositiveFloat = 0.001
    """
    Gradient descent learning rate.
    """
    regularization: NonNegativeFloat = 0.015
    """
    Parameter regularization.
    """
    damping: Damping = 5.0
    """
    Bias damping term.
    """
    range: tuple[float, float] | None = None
    """
    Min/max range of ratings to clamp output.
    """


@dataclass
class FunkSVDTrainingParams:
    learning_rate: float
    regularization: float
    rating_min: float
    rating_max: float


@dataclass
class FunkSVDTrainingData:
    users: pa.Int32Array
    items: pa.Int32Array
    ratings: pa.FloatArray


class FunkSVDScorer(Trainable, Component[ItemList]):
    """
    FunkSVD explicit-feedback matrix factoriation.  FunkSVD is a regularized
    biased matrix factorization technique trained with featurewise stochastic
    gradient descent.

    See the base class :class:`.MFPredictor` for documentation on the estimated
    parameters you can extract from a trained model.

    .. deprecated:: LKPY

        This scorer is kept around for historical comparability, but ALS
        :class:`~lenskit.als.BiasedMF` is usually a better option.

    Stability:
        Caller
    """

    config: FunkSVDConfig

    bias: BiasModel
    users: Vocabulary
    user_embeddings: NPMatrix
    items: Vocabulary
    item_embeddings: NPMatrix

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        """
        Train a FunkSVD model.

        Args:
            ratings: the ratings data frame.
        """
        if hasattr(self, "item_embeddings") and not options.retrain:
            return

        log = _logger.bind(dataset=data.name)

        timer = Stopwatch()
        rmat = data.interactions().matrix().scipy(attribute="rating", layout="coo")

        n_users, n_items = rmat.shape
        n_ratings = rmat.nnz
        log = log.bind(n_users=n_users, n_items=n_items, n_ratings=n_ratings)

        log.info("[%s] fitting bias model", timer)
        self.bias = BiasModel.learn(data, damping=self.config.damping)

        log.info("[%s] preparing rating data", timer)
        log.debug("shuffling rating data")
        shuf = np.arange(n_ratings, dtype=np.int_)
        rng = options.random_generator()
        rng.shuffle(shuf)
        users = pa.array(rmat.row[shuf], pa.int32())
        items = pa.array(rmat.col[shuf], pa.int32())
        ratings = pa.array(rmat.data[shuf], pa.float32())

        del rmat  # don't need it now that data is in Arrow

        log.debug("[%s] computing initial estimates", timer)
        est = np.full(n_ratings, self.bias.global_bias, dtype=np.float32)
        if self.bias.item_biases is not None:
            est += self.bias.item_biases[items.to_numpy()]
        if self.bias.user_biases is not None:
            est += self.bias.user_biases[users.to_numpy()]

        log.debug("[%s] initializing embeddings")
        esize = self.config.embedding_size
        uemb = np.full([n_users, esize], INITIAL_VALUE, dtype=np.float32, order="F")
        iemb = np.full([n_items, esize], INITIAL_VALUE, dtype=np.float32, order="F")

        if self.config.range is not None:
            rmin, rmax = self.config.range
        else:
            rmin = -np.inf
            rmax = np.inf

        config = FunkSVDTrainingParams(
            learning_rate=self.config.learning_rate,
            regularization=self.config.regularization,
            rating_min=rmin,
            rating_max=rmax,
        )
        train_data = FunkSVDTrainingData(users=users, items=items, ratings=ratings)

        trainer = FunkSVDTrainer(config, train_data, uemb, iemb)

        log.info("beginning FunkSVD training")
        with item_progress("FunkSVD dimensions", self.config.embedding_size) as pb:
            for f in range(self.config.embedding_size):
                flog = log.bind(dim=f)
                start = time.perf_counter()
                trail = INITIAL_VALUE * INITIAL_VALUE * (esize - f - 1)
                with item_progress(f"Feature {f + 1} epochs", self.config.epochs) as epb:
                    for e in range(self.config.epochs):
                        elog = flog.bind(epoch=e)
                        rmse = trainer.feature_epoch(f, est, trail)
                        elog.debug("[%s] finished epoch with RMSE %.3f", timer, rmse)
                        epb.update()

                end = time.perf_counter()
                flog.info(
                    "[%s] finished feature %d (RMSE=%f) in %.2fs", timer, f, rmse, end - start
                )
                pb.update()

                est = np.clip(
                    est + uemb[users.to_numpy(), f] * iemb[items.to_numpy(), f], rmin, rmax
                )

        _logger.info("finished model training in %s", timer)

        self.users = data.users
        self.items = data.items
        self.user_embeddings = uemb
        self.item_embeddings = iemb

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        user_id = query.user_id
        user_num = None
        if user_id is not None:
            user_num = self.users.number(user_id, missing=None)
        if user_num is None:
            _logger.debug("unknown user %s", query.user_id)
            return ItemList(items, scores=np.nan)

        u_feat = self.user_embeddings[user_num, :]

        item_nums = items.numbers(vocabulary=self.items, missing="negative")
        item_mask = item_nums >= 0
        i_feats = self.item_embeddings[item_nums[item_mask], :]

        scores = np.full((len(items),), np.nan, dtype=np.float64)
        scores[item_mask] = i_feats @ u_feat
        biases, _ub = self.bias.compute_for_items(items, user_id, query.user_items)
        scores += biases

        return ItemList(items, scores=scores)
