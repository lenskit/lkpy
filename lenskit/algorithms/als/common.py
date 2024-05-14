from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Iterator, NamedTuple, Optional, Self

import numpy as np
import pandas as pd
import torch
from progress_api import make_progress
from seedbank import SeedLike, numpy_rng

from lenskit import util
from lenskit.algorithms.mf_common import MFPredictor
from lenskit.parallel.config import ensure_parallel_init


class TrainContext(NamedTuple):
    """
    Context object for one half of an ALS training operation.
    """

    label: str
    matrix: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    reg: float
    nrows: int
    ncols: int
    embed_size: int
    regI: torch.Tensor

    @classmethod
    def create(
        cls, label: str, matrix: torch.Tensor, left: torch.Tensor, right: torch.Tensor, reg: float
    ) -> TrainContext:
        nrows, ncols = matrix.shape
        lnr, embed_size = left.shape
        assert lnr == nrows
        assert right.shape == (ncols, embed_size)
        regI = torch.eye(embed_size, dtype=left.dtype, device=left.device) * reg
        return TrainContext(label, matrix, left, right, reg, nrows, ncols, embed_size, regI)


class TrainingData(NamedTuple):
    """
    Data for training the ALS model.
    """

    users: pd.Index
    "User ID mapping."
    items: pd.Index
    "Item ID mapping."
    ui_rates: torch.Tensor
    "User-item rating matrix."
    iu_rates: torch.Tensor
    "Item-user rating matrix."

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)

    @classmethod
    def create(cls, users: pd.Index, items: pd.Index, ratings: torch.Tensor) -> TrainingData:
        assert ratings.shape == (len(users), len(items))

        transposed = ratings.transpose(0, 1).to_sparse_csr()
        return cls(users, items, ratings, transposed)

    def to(self, device):
        """
        Move the training data to another device.
        """
        return self._replace(ui_rates=self.ui_rates.to(device), iu_rates=self.iu_rates.to(device))


class ALSBase(MFPredictor):
    """
    Base class for ALS models.
    """

    features: int
    epochs: int
    reg: float | tuple[float, float]
    rng: np.random.Generator
    save_user_features: bool

    user_index_: pd.Index | None
    item_index_: pd.Index
    user_features_: torch.Tensor | None
    item_features_: torch.Tensor

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """
        Overridden in implementation to provide the logger.
        """
        ...

    def __init__(
        self,
        features: int,
        *,
        epochs: int = 10,
        reg: float | tuple[float, float] = 0.1,
        rng_spec: Optional[SeedLike] = None,
        save_user_features: bool = True,
    ):
        self.features = features
        self.epochs = epochs
        self.reg = reg
        self.rng = numpy_rng(rng_spec)
        self.save_user_features = save_user_features

    def fit(self, ratings: pd.DataFrame, **kwargs):
        """
        Run ALS to train a model.

        Args:
            ratings: the ratings data frame.

        Returns:
            The algorithm (for chaining).
        """
        ensure_parallel_init()
        timer = util.Stopwatch()

        for algo in self.fit_iters(ratings, timer=timer, **kwargs):
            pass  # we just need to do the iterations

        if self.user_features_ is not None:
            self.logger.info(
                "trained model in %s (|P|=%f, |Q|=%f)",
                timer,
                torch.norm(self.user_features_, "fro"),
                torch.norm(self.item_features_, "fro"),
            )
        else:
            self.logger.info(
                "trained model in %s (|Q|=%f)",
                timer,
                torch.norm(self.item_features_, "fro"),
            )

        return self

    def fit_iters(
        self, ratings: pd.DataFrame, *, timer: util.Stopwatch | None = None, **kwargs
    ) -> Iterator[Self]:
        """
        Run ALS to train a model, yielding after each iteration.

        Args:
            ratings: the ratings data frame.
        """
        if timer is None:
            timer = util.Stopwatch()

        data = self.prepare_data(ratings)
        self.user_index_ = data.users
        self.item_index_ = data.items

        self.initialize_params(data)

        if isinstance(self.reg, tuple):
            ureg, ireg = self.reg
        else:
            ureg = ireg = self.reg

        assert self.user_features_ is not None
        assert self.item_features_ is not None
        u_ctx = TrainContext.create(
            "user", data.ui_rates, self.user_features_, self.item_features_, ureg
        )
        i_ctx = TrainContext.create(
            "item", data.iu_rates, self.item_features_, self.user_features_, ireg
        )

        self.logger.info(
            "[%s] training biased MF model with ALS for %d features", timer, self.features
        )
        start = timer.elapsed()

        with make_progress(self.logger, "BiasedMF", self.epochs) as epb:
            for epoch in range(self.epochs):
                epoch = epoch + 1

                du = self.als_half_epoch(epoch, u_ctx)
                self.logger.debug("[%s] finished user epoch %d", timer, epoch)

                di = self.als_half_epoch(epoch, i_ctx)
                self.logger.debug("[%s] finished item epoch %d", timer, epoch)

                self.logger.info(
                    "[%s] finished epoch %d (|ΔP|=%.3f, |ΔQ|=%.3f)", timer, epoch, du, di
                )
                epb.update()
                yield self

        if not self.save_user_features:
            self.user_features_ = None
            self.user_index_ = None

        end = timer.elapsed()
        self.logger.info(
            "[%s] trained %d epochs (%.1fs/epoch)",
            timer,
            self.epochs,
            (end - start) / self.epochs,
        )

    @abstractmethod
    def prepare_data(self, ratings: pd.DataFrame) -> TrainingData:
        """
        Prepare data for training this model.  This takes in the ratings, and is
        supposed to do two things:

        -   Normalize or transform the rating/interaction data, as needed, for
            training.
        -   Store any parameters learned from the normalization (e.g. means) in
            the appropriate member variables.
        -   Return the training data object to use for model training.
        """

    def initialize_params(self, data: TrainingData):
        """
        Initialize the model parameters at the beginning of training.
        """
        self.logger.debug("initializing item matrix")
        imat = self.rng.standard_normal((data.n_items, self.features))
        imat = torch.from_numpy(imat)
        self.logger.debug("|Q|: %f", torch.norm(imat, "fro"))
        self.item_features_ = imat

        self.logger.debug("initializing user matrix")
        umat = self.rng.standard_normal((data.n_users, self.features))
        umat = torch.from_numpy(umat)
        self.logger.debug("|P|: %f", torch.norm(umat, "fro"))
        self.user_features_ = umat

    @abstractmethod
    def als_half_epoch(self, epoch: int, context: TrainContext) -> float: ...

    def predict_for_user(self, user, items, ratings: Optional[pd.Series] = None):
        scores = None
        u_offset = None
        u_feat = None
        if ratings is not None and len(ratings) > 0:
            u_feat, u_offset = self.new_user_embedding(user, ratings)

        scores = self.score_by_ids(user, items, u_feat)
        scores = self.finalize_scores(user, scores, u_offset)

        return scores

    @abstractmethod
    def new_user_embedding(self, user, ratings: pd.Series) -> tuple[torch.Tensor, float | None]:
        """
        Generate an embedding for a user given their current ratings.
        """
        ...

    def finalize_scores(self, user, scores: torch.Tensor, u_offset: float | None) -> torch.Tensor:
        """
        Perform any final transformation of scores prior to returning them.
        """
        return scores
