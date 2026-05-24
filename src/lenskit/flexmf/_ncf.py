# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from pydantic import PositiveInt
from torch import Tensor, nn
from torch.linalg import norm

from lenskit.logging import get_logger

from ._base import FlexMFScorerBase
from ._implicit import FlexMFImplicitConfig, FlexMFImplicitTrainer
from ._model import FlexMFModel

_log = get_logger(__name__)


class FlexMFNCFConfig(FlexMFImplicitConfig):
    """
    Configuration for NCF (Neural Collaborative Filtering) Scorer.  It inherits
    common training options and implicit-feedback settings from
    :class:`FlexMFImplicitConfig`.  The inherited ``embedding_size`` field is
    used as the GMF embedding size.

    Stability:
        Experimental
    """

    mlp_embedding_size: PositiveInt = 8
    """
    The size of the MLP embedding space.
    """

    mlp_layers: list[PositiveInt] = [16, 8, 4]
    """
    The sizes of the MLP hidden layers.
    """

    # Override implicit defaults: NCF uses more negatives and no bias terms by default.
    negative_count: PositiveInt = 4
    user_bias: bool | None = False
    item_bias: bool = False


class FlexMFNCFModel(nn.Module):
    """
    Torch module for Neural Collaborative Filtering (NCF).

    Uses composition rather than inheritance: holds a :class:`FlexMFModel` for
    the GMF path alongside a separate MLP tower, combining both via a final
    linear layer.
    """

    gmf_model: FlexMFModel

    def __init__(
        self,
        gmf_e_size: int,
        mlp_e_size: int,
        mlp_layers: list[int],
        n_users: int,
        n_items: int,
        rng: torch.Generator,
        init_scale: float = 0.1,
        sparse: bool = False,
    ):
        super().__init__()

        # GMF component: a standard MF model providing u_embed / i_embed.
        self.gmf_model = FlexMFModel(
            e_size=gmf_e_size,
            n_users=n_users,
            n_items=n_items,
            rng=rng,
            user_bias=False,
            item_bias=False,
            init_scale=init_scale,
            sparse=sparse,
            layers=0,
        )

        self.mlp_e_size = mlp_e_size
        self.u_mlp_embed = nn.Embedding(n_users, mlp_e_size, sparse=sparse)
        self.i_mlp_embed = nn.Embedding(n_items, mlp_e_size, sparse=sparse)

        nn.init.normal_(self.u_mlp_embed.weight, std=init_scale, generator=rng)
        nn.init.normal_(self.i_mlp_embed.weight, std=init_scale, generator=rng)

        # Build MLP
        mlp_modules = []
        input_size = mlp_e_size * 2
        for size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, size))
            mlp_modules.append(nn.ReLU())
            input_size = size

        self.mlp = nn.Sequential(*mlp_modules)

        # Final output layer: concatenated GMF and MLP outputs → scalar score.
        self.prediction = nn.Linear(input_size + gmf_e_size, 1)
        nn.init.kaiming_uniform_(self.prediction.weight, a=1, nonlinearity="sigmoid")

    @property
    def device(self):
        return self.gmf_model.device

    def zero_users(self, users: Tensor):
        """Zero weights for users with no training interactions."""
        self.gmf_model.zero_users(users)
        self.u_mlp_embed.weight.data[users] = 0

    def zero_items(self, items: Tensor):
        """Zero weights for items with no training interactions."""
        self.gmf_model.zero_items(items)
        self.i_mlp_embed.weight.data[items] = 0

    def forward(self, user: Tensor, item: Tensor, *, return_norm: bool = False):
        u_gmf = self.gmf_model.u_embed(user)
        i_gmf = self.gmf_model.i_embed(item)

        u_mlp = self.u_mlp_embed(user)
        i_mlp = self.i_mlp_embed(item)

        # Broadcast MLP embeddings to the same shape before concatenating.
        if u_mlp.shape != i_mlp.shape:
            u_mlp = u_mlp.expand(i_mlp.shape[:-1] + (u_mlp.shape[-1],))
            i_mlp = i_mlp.expand(u_mlp.shape[:-1] + (i_mlp.shape[-1],))

        # GMF path: element-wise product of user and item embeddings.
        gmf_out = u_gmf * i_gmf
        # MLP path: concatenate embeddings and pass through the MLP tower.
        mlp_out = self.mlp(torch.cat([u_mlp, i_mlp], dim=-1))

        # Combine GMF and MLP outputs, then project to a scalar.
        # squeeze(-1) removes the trailing size-1 dimension produced by the
        # linear layer, giving shape (...) instead of (..., 1).
        out = torch.cat([gmf_out, mlp_out], dim=-1)
        score = self.prediction(out).squeeze(-1)

        if return_norm:
            l2 = (
                norm(u_gmf, dim=-1)
                + norm(i_gmf, dim=-1)
                + norm(u_mlp, dim=-1)
                + norm(i_mlp, dim=-1)
            )
            return torch.stack((score, l2))

        return score


class FlexMFNCFTrainer(FlexMFImplicitTrainer):
    """
    Trainer for the NCF Model. Repurposes ImplicitTrainer's loop.
    """

    def create_model(self) -> FlexMFNCFModel:
        return FlexMFNCFModel(
            gmf_e_size=self.config.embedding_size,
            mlp_e_size=self.config.mlp_embedding_size,  # type: ignore
            mlp_layers=self.config.mlp_layers,  # type: ignore
            n_users=self.data.n_users,
            n_items=self.data.n_items,
            rng=self.torch_rng,
            sparse=self.config.reg_method != "AdamW",
        )


class FlexMFNCFScorer(FlexMFScorerBase):
    """
    Neural Collaborative Filtering (NCF) with FlexMF.

    Stability:
        Experimental
    """

    config: FlexMFNCFConfig

    def create_trainer(self, data, options):
        return FlexMFNCFTrainer(self, data, options)
