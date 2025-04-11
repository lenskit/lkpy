# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from typing_extensions import override

from lenskit.data import Dataset
from lenskit.flexmf._model import FlexMFModel

from ._base import FlexMFConfigBase, FlexMFScorerBase
from ._training import (
    FlexMFTrainerBase,
    FlexMFTrainingBatch,
    FlexMFTrainingData,
)


@dataclass
class FlexMFExplicitConfig(FlexMFConfigBase):
    """
    Configuration for :class:`FlexMFExplicitScorer`.

    Stability:
        Experimental
    """


class FlexMFExplicitScorer(FlexMFScorerBase):
    """
    Explicit-feedback rating prediction with FlexMF.  This realizes a biased
    matrix factorization model (similar to :class:`lenskit.als.BiasedMF`)
    trained with PyTorch.

    Stability:
        Experimental
    """

    global_bias: float

    def create_trainer(self, data, options):
        return FlexMFExplicitTrainer(self, data, options)

    def score_items(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return super().score_items(users, items) + self.global_bias


class FlexMFExplicitTrainer(FlexMFTrainerBase[FlexMFExplicitScorer]):
    @override
    def prepare_data(self, data: Dataset) -> FlexMFTrainingData:
        """
        Set up the training data and context for the scorer.
        """

        ratings = data.interaction_matrix(field="rating", format="torch", layout="coo")

        # extract the components: row, column, value.
        # indices are a matrix in Torch COO format.
        indices = ratings.indices()
        rm_users = indices[0, :]
        rm_items = indices[1, :]
        rm_values = ratings.values().to(torch.float32)

        # compute the global mean (global bias), and subtract from all rating values
        mean = rm_values.mean()
        rm_values = rm_values - mean

        # save data we learned at this stage
        self.component.global_bias = mean.item()
        self.component.users = data.users
        self.component.items = data.items

        return FlexMFTrainingData(
            batch_size=self.config.batch_size,
            n_users=data.user_count,
            n_items=data.item_count,
            users=rm_users,
            items=rm_items,
            fields={"ratings": rm_values},
        ).to(self.device)

    @override
    def create_model(self) -> FlexMFModel:
        """
        Prepare the model for training.
        """
        return FlexMFModel(
            self.config.embedding_size,
            self.data.n_users,
            self.data.n_items,
            self.torch_rng,
            sparse=self.config.reg_method != "AdamW",
            init_scale=0.1,
        )

    @override
    def train_batch(self, batch: FlexMFTrainingBatch) -> float:
        if self.config.reg_method == "L2":
            result = self.fast_model(batch.users, batch.items, return_norm=True)
            pred = result[0, :]
            norm = torch.mean(result[1, :]) * self.config.regularization
        else:
            pred = self.fast_model(batch.users, batch.items)
            norm = 0.0

        loss = F.mse_loss(pred, batch.fields["ratings"])

        loss_all = loss + norm

        loss_all.backward()
        self.opt.step()

        return loss.item()
