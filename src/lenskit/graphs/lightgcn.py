# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LightGCN recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pydantic import PositiveFloat, PositiveInt, model_validator
from structlog.stdlib import BoundLogger
from torch_geometric.nn import LightGCN
from typing_extensions import Literal, Self

from lenskit import logging
from lenskit.data import BatchIter, Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.data.matrix import COOStructure
from lenskit.data.relationships import MatrixRelationshipSet
from lenskit.logging.progress._dispatch import item_progress
from lenskit.pipeline.components import Component
from lenskit.torch import inference_mode
from lenskit.training import ModelTrainer, TrainingOptions, UsesTrainer

_log = logging.get_logger(__name__)


@dataclass
class LightGCNConfig:
    """
    Configuration for :class:`LightGCNScorer`.

    Stability:
        Experimental
    """

    embedding_size: PositiveInt = 50
    """
    The dimension of the embedding space (number of latent features).
    """

    layer_count: PositiveInt = 2
    """
    The number of layers to use.
    """

    layer_blend: PositiveFloat | list[PositiveFloat] | None = None
    """
    The blending coefficient(s) for layer blending.  This is equivalent to
    ``alpha`` in :class:`LightGCN`.
    """

    batch_size: PositiveInt = 8 * 1024
    """
    The training batch size.
    """

    learning_rate: PositiveFloat = 0.01
    """
    The learning rate for training.
    """

    epochs: PositiveInt = 10
    """
    The number of training epochs.
    """

    regularization: PositiveFloat | None = 0.1
    """
    The regularization strength.
    """

    loss: Literal["logistic", "pairwise"] = "pairwise"
    """
    The loss to use for model training.

    ``pairwise``
        BPR pairwise ranking loss, using :meth:`LightGCN.recommend_loss`.

    ``logistic``
        Logistic link prediction loss, using :meth:`LightGCN.link_pred_loss`.
    """

    negative_count: PositiveInt = 1
    """
    The number of negative items to sample for each positive item in the
    training data.
    """

    @model_validator(mode="after")
    def check_layer_blending(self) -> Self:
        if isinstance(self.layer_blend, list) and len(self.layer_blend) != self.layer_count:
            raise ValueError(
                f"layer_blend has length {len(self.layer_blend)}, expected {self.layer_count}"
            )

        return self


class LightGCNScorer(UsesTrainer, Component[ItemList]):
    """
    Scorer using :class:`LightGCN` :cite:p:`heLightGCNSimplifyingPowering2020`.
    """

    config: LightGCNConfig

    users: Vocabulary
    items: Vocabulary
    model: LightGCN
    _user_base: int
    """
    Offset for the user nodes in the data set. Item edges start from zero.
    """
    _edges: torch.Tensor
    """
    The graph edges as used by the graph model.
    """

    def to(self, device):
        "Move the model to a different device."
        self.model = self.model.to(device)
        self._edges = self._edges.to(device)
        return self

    @inference_mode
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        """
        Generate item scores for a user.

        Note that user and items are both user and item IDs, not positions.
        """
        # make sure the query is in a known / usable format
        query = RecQuery.create(query)

        # if we have no user ID, we cannot score items
        u_row = None
        if query.user_id is not None:
            u_row = self.users.number(query.user_id, missing=None)

        if u_row is None:
            return ItemList(items, scores=np.nan)

        # look up the item columns in the embedding matrix
        i_cols = items.numbers(vocabulary=self.items, missing="negative", format="torch")
        i_cols = i_cols.to(self._edges.device)

        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable_mask = i_cols.ge(0)
        i_cols = i_cols.masked_select(scorable_mask)

        # set up the edge tensor
        u_tensor = torch.from_numpy(np.repeat([u_row + self._user_base], len(i_cols)))
        u_tensor = u_tensor.to(self._edges.device)
        edges = torch.stack([u_tensor, i_cols])
        scores = self.model(self._edges, edges)

        # initialize output score array, fill with missing
        full_scores = torch.full((len(items),), np.nan, dtype=torch.float32, device=scores.device)
        full_scores.masked_scatter_(scorable_mask, scores)

        # return the result!
        return ItemList(items, scores=full_scores)

    def create_trainer(self, data, options):
        match self.config.loss:
            case "logistic":
                return LogisticLightGCNTrainer(self, data, options)
            case "pairwise":
                return PairwiseLightGCNTrainer(self, data, options)
            case _:  # pragam: nocover
                raise ValueError("invalid loss")


class LightGCNTrainer(ModelTrainer):
    scorer: LightGCNScorer
    data: Dataset
    options: TrainingOptions
    log: BoundLogger

    rng: np.random.Generator
    device: str
    model: LightGCN

    matrix: MatrixRelationshipSet
    coo: COOStructure
    user_base: int
    edges: torch.Tensor

    optimizer: torch.optim.Optimizer
    epochs_trained: int = 0

    def __init__(self, scorer: LightGCNScorer, data: Dataset, options: TrainingOptions):
        self.scorer = scorer
        self.data = data
        self.options = options
        self.device = options.configured_device(gpu_default=True)
        self.log = _log.bind()
        self.rng = options.random_generator()

        self.user_base = data.item_count
        node_count = data.user_count + data.item_count
        if isinstance(scorer.config.layer_blend, list):
            blend = torch.tensor(scorer.config.layer_blend)
        else:
            blend = scorer.config.layer_blend

        self.matrix = data.interactions().matrix()
        self.coo = coo = self.matrix.coo_structure()
        e_src = torch.tensor(coo.row_numbers + self.user_base)
        e_dst = torch.tensor(coo.col_numbers)
        self.edges = torch.stack([e_src, e_dst]).to(self.device)
        print(self.coo)

        self.model = LightGCN(
            node_count, scorer.config.embedding_size, scorer.config.layer_count, blend
        ).to(self.device)

        if scorer.config.regularization:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                scorer.config.learning_rate,
                weight_decay=scorer.config.regularization,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                scorer.config.learning_rate,
            )

        self.scorer.items = self.data.items
        self.scorer.users = self.data.users
        self.scorer.model = self.model
        self.scorer._user_base = self.user_base
        self.scorer._edges = self.edges

    def train_epoch(self) -> dict[str, float] | None:
        epoch = self.epochs_trained + 1
        self.log = elog = self.log.bind(epoch=epoch)
        elog.debug("creating epoch training data")
        config = self.scorer.config

        # permute the data
        perm = np.require(self.rng.permutation(self.coo.nnz), dtype=np.int32)
        perm_t = torch.from_numpy(perm).to(self.device)

        batches = BatchIter(self.coo.nnz, config.batch_size)

        tot_loss = torch.tensor(0.0).to(self.device)
        avg_loss = np.nan
        with item_progress(f"Training epoch {epoch}", len(batches), {"loss": ".3f"}) as pb:
            elog.debug("beginning epoch")
            for i, (bs, be) in enumerate(batches, start=1):
                self.log = blog = elog.bind(batch=i)
                blog.debug("training batch")

                pos = self.edges[:, perm_t[bs:be]]
                neg = self.matrix.sample_negatives(
                    self.coo.row_numbers[perm[bs:be]],
                    n=config.negative_count,
                    rng=self.rng,
                )
                neg = torch.from_numpy(neg)
                neg = neg.to(self.device, non_blocking=True)
                neg = torch.stack([pos[0], neg[:, 0]])

                mb_edges = torch.cat([pos, neg], 1)
                assert mb_edges.shape == (2, (be - bs) * 2)

                scores = self.model(self.edges, mb_edges)
                loss = self.batch_loss(mb_edges, scores)

                loss.backward()
                self.optimizer.step()

                if i % 20 == 0:
                    avg_loss = tot_loss.item() / i
                pb.update(loss=avg_loss)
                tot_loss += loss

        avg_loss = tot_loss.item() / len(batches)
        elog.debug("epoch complete", loss=avg_loss)
        self.epochs_trained += 1
        return {"loss": avg_loss}

    def finalize(self):
        pass

    def batch_loss(self, mb_edges: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class LogisticLightGCNTrainer(LightGCNTrainer):
    def batch_loss(self, mb_edges: torch.Tensor, scores: torch.Tensor):
        (n,) = scores.shape
        labels = torch.from_numpy(np.repeat([1, 0], n // 2))
        return self.model.link_pred_loss(scores, labels)


class PairwiseLightGCNTrainer(LightGCNTrainer):
    def batch_loss(self, mb_edges: torch.Tensor, scores: torch.Tensor):
        (n,) = scores.shape
        pos_score, neg_score = scores.chunk(2)
        # FIXME: set up better regularization
        return self.model.recommendation_loss(pos_score, neg_score, node_id=mb_edges.ravel())
