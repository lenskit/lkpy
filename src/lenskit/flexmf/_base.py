# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from lenskit.data import ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.logging import get_logger
from lenskit.pipeline import Component
from lenskit.torch import inference_mode
from lenskit.training import UsesTrainer

from ._model import FlexMFModel

# I want a logger for information
_log = get_logger(__name__)


@dataclass
class FlexMFConfigBase:
    """
    Common configuration for all FlexMF scoring components.

    Stability:
        Experimental
    """

    embedding_size: int = 50
    """
    The dimension of the embedding space (number of latent features).
    """

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

    regularization: float = 0.1
    """
    The regularization strength.
    """

    reg_method: Literal["AdamW", "L2"] | None = "AdamW"
    """
    The regularization method to use.

    With the default AdamW regularization, training will use the
    :class:`~torch.optim.AdamW` optimizer with weight decay. With L2
    regularization, training will use sparse gradients and the
    :class:`torch.optim.SparseAdam` optimizer.

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

        # if we have no user ID, we cannot score items
        # TODO: support pooling from user history
        u_row = None
        if query.user_id is not None:
            u_row = self.users.number(query.user_id, missing=None)

        if u_row is None:
            return ItemList(items, scores=np.nan)

        # look up the user row in the embedding matrix
        u_tensor = torch.IntTensor([u_row])
        # make sure it's on the right device
        device = self.model.device
        u_tensor = u_tensor.to(device, non_blocking=True)

        # look up the item columns in the embedding matrix
        i_cols = items.numbers(vocabulary=self.items, missing="negative", format="torch")
        i_cols = i_cols.to(device, non_blocking=True)

        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable_mask = i_cols.ge(0)
        i_cols = i_cols.masked_select(scorable_mask)

        # get scores
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
