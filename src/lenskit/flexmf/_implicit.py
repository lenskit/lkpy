# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch
from torch.nn import functional as F

from lenskit.data import Dataset

from ._base import FlexMFConfigBase, FlexMFScorerBase
from ._model import FlexMFModel
from ._training import FlexMFTrainerBase, FlexMFTrainingBatch, FlexMFTrainingData

ImplicitLoss: TypeAlias = Literal["logistic", "pairwise"]
NegativeStrategy: TypeAlias = Literal["uniform", "popular"]


@dataclass
class FlexMFImplicitConfig(FlexMFConfigBase):
    """
    Configuration for :class:`FlexMFImplicitScorer`.  It inherits base model
    options from :class:`FlexMFConfigBase`.

    Stability:
        Experimental
    """

    loss: ImplicitLoss = "logistic"
    """
    The loss to use for model training.
    """

    negative_strategy: NegativeStrategy = "uniform"
    """
    The negative sampling strategy.
    """

    negative_count: int = 1
    """
    The number of negative items to sample for each positive item in the training data.
    """

    positive_weight: float = 1.0
    """
    A weighting multiplier to apply to the positive item's loss, to adjust the
    relative importance of positive and negative classifications.  Only applies
    to logistic loss.
    """

    user_bias: bool | None = None
    """
    Whether to learn a user bias term.  If unspecified, the default depends on the
    loss function (``False`` for pairwise and ``True`` for logistic).
    """
    item_bias: bool = True
    """
    Whether to learn an item bias term.
    """


class FlexMFImplicitScorer(FlexMFScorerBase):
    """
    Implicit-feedback rating prediction with FlexMF.  This is capable of
    realizing multiple models, including:

    - BPR-MF (Bayesian personalized ranking) :cite:p:`BPR` (with ``"pairwise"`` loss)
    - Logistic matrix factorization :cite:p:`LogisticMF` (with ``"logistic"`` loss)

    All use configurable negative sampling, including the sampling approach from WARP.

    Stability:
        Experimental
    """

    config: FlexMFImplicitConfig

    def create_trainer(self, data, options):
        return FlexMFImplicitTrainer(self, data, options)


class FlexMFImplicitTrainer(FlexMFTrainerBase[FlexMFImplicitScorer, FlexMFImplicitConfig]):
    def prepare_data(self, data: Dataset) -> FlexMFTrainingData:
        """
        Set up the training data and context for the scorer.
        """

        matrix = data.interactions().matrix()
        coo = matrix.coo_structure()

        # save data we learned at this stage
        self.component.users = data.users
        self.component.items = data.items

        return FlexMFTrainingData(
            batch_size=self.config.batch_size,
            n_users=data.user_count,
            n_items=data.item_count,
            users=coo.row_numbers,
            items=coo.col_numbers,
            matrix=matrix,
        )

    def create_model(self) -> FlexMFModel:
        """
        Prepare the model for training.
        """
        user_bias = self.config.user_bias
        if user_bias is None:
            if self.config.loss == "pairwise":
                user_bias = False
            else:
                user_bias = True

        return FlexMFModel(
            self.config.embedding_size,
            self.data.n_users,
            self.data.n_items,
            self.torch_rng,
            user_bias=user_bias,
            item_bias=self.config.item_bias,
            sparse=self.config.reg_method != "AdamW",
        )

    def train_batch(self, batch: FlexMFTrainingBatch) -> float:
        assert batch.data.matrix is not None
        assert isinstance(batch.users, np.ndarray)
        negatives = batch.data.matrix.sample_negatives(
            batch.users,
            weighting=self.config.negative_strategy,
            n=self.config.negative_count,
            rng=self.rng,
        )

        users = torch.as_tensor(batch.users.reshape(-1, 1)).to(self.device)
        positives = torch.as_tensor(batch.items.reshape(-1, 1))
        negatives = torch.as_tensor(negatives)
        items = torch.cat((positives, negatives), 1).to(self.device)

        if self.config.reg_method == "L2":
            result = self.model(users, items, return_norm=True)
            # :1 instead of 0 to reduce shape-adjustment overhead
            pos_pred = result[0, :, :1]
            neg_pred = result[0, :, 1:]

            norm = torch.mean(result[1, ...]) * self.config.regularization
        else:
            result = self.model(users, items)
            pos_pred = result[:, :1]
            neg_pred = result[:, 1:]
            norm = 0.0

        match self.config.loss:
            case "logistic":
                pos_lp = -F.logsigmoid(pos_pred) * self.config.positive_weight
                neg_lp = -F.logsigmoid(-neg_pred)
                tot_lp = pos_lp.sum() + neg_lp.sum()
                tot_n = pos_lp.nelement() + neg_lp.nelement()
                loss = tot_lp / tot_n

            case "pairwise":
                lp = -F.logsigmoid(pos_pred - neg_pred)
                loss = lp.mean()

        loss_all = loss + norm

        loss_all.backward()
        self.opt.step()

        return loss.item()
