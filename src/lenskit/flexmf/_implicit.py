# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
import torch
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt, model_validator
from torch.nn import functional as F
from typing_extensions import override

from lenskit.data import Dataset
from lenskit.logging import get_logger

from ._base import FlexMFConfigBase, FlexMFScorerBase
from ._model import FlexMFModel
from ._training import FlexMFTrainerBase, FlexMFTrainingBatch, FlexMFTrainingData

WARP_CAND_BATCH_SIZE = 10
MAX_TRIES = 200
ImplicitLoss: TypeAlias = Literal["logistic", "pairwise", "warp"]
NegativeStrategy: TypeAlias = Literal["uniform", "popular", "misranked"]

_log = get_logger(__name__)

PRESETS = {
    "bpr": {"loss": "pairwise", "user_bias": False, "item_bias": False},
    "warp": {
        "loss": "warp",
        "negative_strategy": "misranked",
        "user_bias": False,
        "item_bias": False,
    },
    "lightgcn": {
        "loss": "pairwise",
        "user_bias": False,
        "item_bias": False,
        "convolution_layers": 3,
    },
}


class FlexMFImplicitConfig(FlexMFConfigBase):
    """
    Configuration for :class:`FlexMFImplicitScorer`.  It inherits base model
    options from :class:`FlexMFConfigBase`.

    Stability:
        Experimental
    """

    preset: Literal["bpr", "warp", "lightgcn"] | None = None
    """
    Select preset defaults to mimic a particular model's original presentation.
    """

    loss: ImplicitLoss = "logistic"
    """
    The loss to use for model training.
    """

    negative_strategy: NegativeStrategy | None = None
    """
    The negative sampling strategy.  The default is ``"misranked"`` for WARP
    loss and ``"uniform"`` for other losses.
    """

    negative_count: PositiveInt = 1
    """
    The number of negative items to sample for each positive item in the
    training data.  With BPR loss, the positive item is compared to each
    negative item; with logistic loss, the positive item is treated once per
    learning round, so this setting effectively makes the model learn on _n_
    negatives per positive, rather than giving positive and negative examples
    equal weight.
    """

    positive_weight: PositiveFloat = 1.0
    """
    A weighting multiplier to apply to the positive item's loss, to adjust the
    relative importance of positive and negative classifications.  Only applies
    to logistic loss.
    """

    user_bias: bool | None = None
    """
    Whether to learn a user bias term.  If unspecified, the default depends on
    the loss function (``False`` for pairwise and ``True`` for logistic).
    """
    item_bias: bool = True
    """
    Whether to learn an item bias term.
    """

    convolution_layers: NonNegativeInt = 0
    """
    The number of LightGCN convolution layers to use.  0 (the default)
    configures for standard matrix factorization.
    """

    def selected_negative_strategy(self) -> NegativeStrategy:
        if self.negative_strategy is not None:
            return self.negative_strategy
        elif self.loss == "warp":
            return "misranked"
        else:
            return "uniform"

    @model_validator(mode="before")
    @classmethod
    def apply_preset(cls, data):
        if preset := data.get("preset", None):
            if preset in PRESETS:
                return PRESETS[preset] | data
            else:
                raise ValueError(f"unknown preset '{preset}'")
        else:
            return data

    @model_validator(mode="after")
    def check_strategies(self):
        if (
            self.loss == "warp"
            and self.negative_strategy is not None
            and self.negative_strategy != "misranked"
        ):
            raise ValueError("WARP loss requires “misranked” negative strategy")

        if self.selected_negative_strategy() == "misranked" and self.negative_count > 1:
            raise ValueError("misrank negatives only works with single negatives")

        return self


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
        if self.config.selected_negative_strategy() == "misranked":
            return FlexMFWARPTrainer(self, data, options)
        else:
            return FlexMFImplicitTrainer(self, data, options)


class FlexMFImplicitTrainer(FlexMFTrainerBase[FlexMFImplicitScorer, FlexMFImplicitConfig]):
    _loss: Callable[[torch.Tensor, torch.Tensor, float, torch.Tensor | None], torch.Tensor]

    def __init__(self, component, data, options):
        super().__init__(component, data, options)
        match self.config.loss:
            case "logistic":
                self._loss = _loss_logistic
            case "pairwise":
                self._loss = _loss_pairwise
            case "warp":
                self._loss = _loss_warp
            case other:  # pragma: nocover
                raise ValueError(f"unknown loss {other}")

    def prepare_data(self, data: Dataset) -> FlexMFTrainingData:
        """
        Set up the training data and context for the scorer.
        """

        matrix = data.interactions().matrix()
        coo = matrix.coo_structure()

        ui_mat = iu_mat = None
        if self.config.convolution_layers:
            _log.debug("preparing convolution neighborhood matrix")
            mat = matrix.torch(layout="coo")
            unorms = mat.sum(1).to_dense().sqrt()
            inorms = mat.sum(0).to_dense().sqrt()
            idx = mat.indices()
            vals = mat.values()
            ui_mat = torch.sparse_coo_tensor(
                idx,
                vals / unorms[idx[0, :]] / inorms[idx[1, :]],
                mat.size(),
                requires_grad=False,
            ).to(self.device)
            iu_mat = ui_mat.T.to_sparse_csr().detach()
            ui_mat = ui_mat.to_sparse_csr().detach()

        # save data we learned at this stage
        self.component.users = data.users
        self.component.items = data.items

        return FlexMFTrainingData(
            batch_size=self.config.batch_size,
            n_users=data.user_count,
            n_items=data.item_count,
            users=coo.row_numbers,
            items=coo.col_numbers,
            interactions=matrix,
            ui_matrix=ui_mat,
            iu_matrix=iu_mat,
        )

    def create_model(self) -> FlexMFModel:
        """
        Prepare the model for training.
        """
        user_bias = self.config.user_bias
        if user_bias is None:
            if self.config.loss == "logistic":
                user_bias = True
            else:
                user_bias = False

        return FlexMFModel(
            self.config.embedding_size,
            self.data.n_users,
            self.data.n_items,
            self.torch_rng,
            user_bias=user_bias,
            item_bias=self.config.item_bias,
            layers=self.config.convolution_layers,
            sparse=self.config.reg_method != "AdamW",
        )

    def score(self, users, items) -> tuple[torch.Tensor, torch.Tensor]:
        if self.explicit_norm:
            result = self.model(users, items, return_norm=True)
            scores = result[0, ...]

            norms = result[1, ...]
        else:
            scores = self.call_model(users, items, return_norm=False)
            norms = torch.tensor(0.0)

        return scores, norms

    def train_batch(self, batch: FlexMFTrainingBatch) -> torch.Tensor:
        # for LightGCN, we have to update the convolution layers *every* batch
        if batch.data.ui_matrix is not None:
            assert batch.data.iu_matrix is not None
            self.model.update_convolution(batch.data.ui_matrix, batch.data.iu_matrix)

        users = torch.as_tensor(batch.users.reshape(-1, 1)).to(self.device, non_blocking=True)
        positives = torch.as_tensor(batch.items.reshape(-1, 1)).to(self.device, non_blocking=True)

        pos_pred, pos_norm = self.score(users, positives)

        neg_items, neg_pred, neg_norm, weights = self.scored_negatives(batch, users, pos_pred)

        loss = self._loss(pos_pred, neg_pred, self.config.positive_weight, weights)

        if self.explicit_norm:
            loss = loss + self.config.regularization * 0.5 * (pos_norm.mean() + neg_norm.mean())

        loss.backward()
        self.opt.step()

        return loss

    def scored_negatives(
        self, batch: FlexMFTrainingBatch, users: torch.Tensor, pos_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert batch.data.interactions is not None
        assert isinstance(batch.users, np.ndarray)
        items = torch.as_tensor(
            batch.data.interactions.sample_negatives(
                batch.users,
                weighting=self.config.selected_negative_strategy(),
                n=self.config.negative_count,
                rng=self.rng,
            )
        ).to(users.device, non_blocking=True)
        scores, norms = self.score(users, items)
        return items, scores, norms, None


class FlexMFWARPTrainer(FlexMFImplicitTrainer):
    @override
    def scored_negatives(
        self, batch: FlexMFTrainingBatch, users: torch.Tensor, pos_scores: torch.Tensor
    ):
        assert batch.data.interactions is not None
        assert isinstance(batch.users, np.ndarray)
        bsize = len(users)
        assert pos_scores.shape == (bsize, 1)
        device = users.device

        # Pre-allocate an array of batch row indices to speed computation.
        idx_range = torch.arange(len(users))
        idx_dev = idx_range.to(device=users.device, non_blocking=True)

        # Allocate tensors to store the final sampled negatives and their
        # scores, and the number of tries we needed to find them.
        neg_scores = torch.full((len(users),), -math.inf, device=device)
        neg_norms = torch.zeros(len(users), device=device)
        neg_counts = torch.zeros(len(users))
        neg_items = torch.empty(len(users), dtype=torch.int32, device=device)

        # The main logic is to loop, sampling candidate negatives and testing
        # them for misranks. MAX_TRIES is the maximum number of candidate
        # negaties we will consider before giving up. We sample and score
        # candidate negatives in batches, of size WARP_CAND_BATCH_SIZE, to
        # reduce the CPU/GPU communication overhead.

        # Data structures to track the *candidate* batch - the set of items
        # we've found based on the users who needed negatives at the start of
        # the candidate batch. These will not be assigned until the first time
        # in the loop.
        cand_mask: torch.Tensor  # training batch mask for rows w/ candidates
        cand_range: torch.Tensor  # index range of candidate rows
        cand_items: torch.Tensor  # candidate item numbers
        cand_scores: torch.Tensor  # candidate item scores
        cand_norms: torch.Tensor  # candidate item norms

        # Count the number of attempted negatives.
        tries = 0
        # Track which training batch rows we still need negatives for.
        needed = torch.full((bsize,), True, dtype=torch.bool)
        n_dev = needed.to(device, non_blocking=True)

        # We will keep looping as long as some rows need negatives, and we
        # haven't exceeded the attempt budget.
        while tries < MAX_TRIES and torch.any(needed):
            # Compute the index within the current batch.
            bi = tries % WARP_CAND_BATCH_SIZE
            tries += 1

            if bi == 0:
                # Set up candidates based on rows that need items at the start of the batch.
                cand_mask = needed
                cand_size = torch.sum(needed).item()
                cand_range = torch.arange(cand_size).to(device=device, non_blocking=True)
                cand_items = torch.as_tensor(
                    batch.data.interactions.sample_negatives(
                        batch.users[needed],
                        n=WARP_CAND_BATCH_SIZE,
                        weighting="uniform",
                        rng=self.rng,
                    )
                ).to(device, non_blocking=True)
                cand_scores, cand_norms = self.score(users[n_dev], cand_items)

            # Which of the *candidate* rows represent needed negatives?
            cand_needed = needed[cand_mask]
            cn_dev = cand_needed.to(device=device, non_blocking=True)

            # For any batch row still needing a negative, if we have a higher score,
            # we've either found that negative, or we've found a better negative for
            # when we give up. So update all the rows in ‘needed’ for which the candidate
            # item score is higher.
            act_better = cand_scores[cn_dev, bi] > neg_scores[n_dev]
            act_cidx = cand_range[cn_dev][act_better]
            # Compute the indices in the batch for these improvements.
            upd_idx = idx_dev[n_dev][act_better]
            # Apply the updates to the training batch working storage.
            neg_counts[upd_idx] = tries
            neg_items[upd_idx] = cand_items[act_cidx, bi]
            neg_scores[upd_idx] = cand_scores[act_cidx, bi]
            if cand_norms.shape:
                neg_norms[upd_idx] = cand_norms[act_cidx, bi]

            # Now that we have the best negative so far, find the winners and
            # remove don't update them in the next batch.
            n_dev = neg_scores < pos_scores[:, 0]
            needed = n_dev.cpu()

        # Compute estimated ranks from the negative counts
        ranks = ((self.data.n_items - 1) / (neg_counts + 1)).to(torch.float64).detach()
        # Used estimated ranks to compute sample weights for the minibatch
        # L(k) = sum i=1..k 1/i = harmonic k
        # approximate harmonic k with log
        weights = (
            torch.log(ranks)
            + np.euler_gamma
            + 1 / (2 * ranks)
            - 1 / (12 * ranks**2)
            + 1 / (120 * ranks**4)
        )
        weights = weights.to(pos_scores.device, non_blocking=True)
        return neg_items, neg_scores.reshape(-1, 1), neg_norms, weights


def _loss_logistic(pos_pred, neg_pred, pos_weight, weights):
    pos_lp = -F.logsigmoid(pos_pred) * pos_weight
    neg_lp = -F.logsigmoid(-neg_pred)
    tot_lp = pos_lp.sum() + neg_lp.sum()
    tot_n = pos_lp.nelement() + neg_lp.nelement()
    return tot_lp / tot_n


def _loss_pairwise(pos_pred, neg_pred, pos_weight, weights):
    lp = -F.logsigmoid(pos_pred - neg_pred)
    return lp.mean()


def _loss_warp(pos_pred, neg_pred, pos_weight, weights):
    assert weights is not None
    lp = -F.logsigmoid(pos_pred - neg_pred) * weights
    return lp.mean()
