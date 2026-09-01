# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, PositiveInt

from lenskit.config.common import EmbeddingSizeMixin
from lenskit.data import ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.logging import get_logger
from lenskit.pipeline import Component
from lenskit.torch import inference_mode
from lenskit.training import UsesTrainer

from ._model import FlexMFModel

# try to import Dynamo to prevent import problems on Torch 2.12
try:
    import torch._dynamo  # noqa: F401
except ImportError:
    pass

# I want a logger for information
_log = get_logger(__name__)


class FlexMFConfigBase(EmbeddingSizeMixin, BaseModel):
    """
    Common configuration for all FlexMF scoring components.

    Stability:
        Experimental
    """

    embedding_size: PositiveInt = 64

    batch_size: int = 8 * 1024
    """
    The training batch size.
    """

    learning_rate: float = 0.01
    """
    The learning rate for training.
    """

    epochs: int = 10
    """
    The number of training epochs.
    """

    regularization: float = 0.01
    """
    The regularization strength.

    .. note::
        The explicit-feedback model uses a different default strength.
    """

    reg_method: Literal["AdamW", "L2"] | None = "AdamW"
    """
    The regularization method to use.

    With the default AdamW regularization, training will use the
    :class:`~torch.optim.AdamW` optimizer with weight decay. With L2
    regularization, training will use sparse gradients and the
    :class:`torch.optim.SparseAdam` optimizer.

    .. note::
        The explicit-feedback model defaults this setting to ``"L2"``.

    ``None``
        Use no regularization.

    ``"L2"``
        Use L2 regularization on the parameters used in each training batch. The
        strength is applied to the _mean_ norms in a batch, so that the
        regularization term scale is not dependent on the batch size.

    ``"AdamW"``
        Use :class:`torch.optim.AdamW` with the specified regularization
        strength.  This configuration does *not* use sparse gradients, but
        training time is often comparable.

    .. note::
        Regularization values do not necessarily have the same range or meaning
        for the different regularization methods.
    """

    user_embeddings: bool | Literal["prefer"] = True
    """
    Whether to use trained user embeddings for scoring.  If ``True``, trained
    embeddings are used when the query does not provide usable query items. If
    ``False``, trained user embeddings are not used for scoring. If set to
    ``"prefer"``, the trained embedding is used for known users even when the
    query provides items.
    """


class FlexMFScorerBase(UsesTrainer, Component):
    """
    Base class for the FlexMF scorers, providing common Torch support.

    Stability:
        Experimental
    """

    config: FlexMFConfigBase
    users: Vocabulary
    items: Vocabulary
    model: FlexMFModel

    def to(self, device):
        "Move the model to a different device."
        self.model = self.model.to(device)
        return self

    @inference_mode
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        """
        Generate item scores for a user.

        Note that user and items are both user and item IDs, not positions.
        """
        # make sure the query is in a known / usable format
        query = RecQuery.create(query)

        # resolve the user, if it is known
        u_row = None
        if query.user_id is not None:
            u_row = self.users.number(query.user_id, missing=None)

        # make sure it's on the right device
        device = self.model.device
        u_tensor = None
        if u_row is not None:
            # look up the user row in the embedding matrix
            u_tensor = torch.IntTensor([u_row])
            u_tensor = u_tensor.to(device, non_blocking=True)

        # decide whether query items should provide the user embedding
        query_items = query.query_items
        pool_query = (
            query_items is not None
            and len(query_items) > 0
            and (u_row is None or self.config.user_embeddings != "prefer")
        )

        pooled_user = None
        if pool_query:
            # resolve query items against the model's item vocabulary
            q_cols = query_items.numbers(vocabulary=self.items, missing="negative", format="torch")
            q_cols = q_cols.to(device, non_blocking=True)

            # ignore query items that were not known during model training
            q_cols = q_cols.masked_select(q_cols.ge(0))

            if len(q_cols) > 0:
                # mean-pool the item embeddings to obtain a query-time
                # representation of the user
                q_vectors = self.model.i_embed(q_cols)
                pooled_user = q_vectors.mean(dim=0)

        # if pooling was not possible, fall back to the trained user embedding
        if pooled_user is None and (u_tensor is None or not self.config.user_embeddings):
            return ItemList(items, scores=np.nan)

        # look up the item columns in the embedding matrix
        i_cols = items.numbers(vocabulary=self.items, missing="negative", format="torch")
        i_cols = i_cols.to(device, non_blocking=True)

        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable_mask = i_cols.ge(0)
        i_cols = i_cols.masked_select(scorable_mask)

        # get scores
        if pooled_user is not None:
            scores = self.score_user_embedding(pooled_user, i_cols)
        else:
            assert u_tensor is not None
            scores = self.score_items(u_tensor, i_cols)

        # initialize output score array, fill with missing
        full_scores = torch.full((len(items),), np.nan, dtype=torch.float32, device=scores.device)
        full_scores.masked_scatter_(scorable_mask, scores)

        # return the result!
        return ItemList(items, scores=full_scores)

    def score_items(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Score for users and items, after resolivng them and limiting to known
        users and items.
        """
        return self.model(users, items)

    def score_user_embedding(self, user: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Score items against a user embedding.
        """
        return self.model.score_user_vector(user, items)
