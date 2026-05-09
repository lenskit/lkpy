# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Literal

import torch
from pydantic import PositiveInt
from torch import Tensor, nn
from torch.linalg import norm

from lenskit.logging import get_logger

from ._base import FlexMFConfigBase, FlexMFScorerBase
from ._model import FlexMFModel
from ._implicit import ImplicitLoss, NegativeStrategy, FlexMFImplicitTrainer

_log = get_logger(__name__)


class FlexMFNCFConfig(FlexMFConfigBase):
    """
    Configuration for NCF (Neural Collaborative Filtering) Scorer.
    
    Stability:
        Experimental
    """
    gmf_embedding_size: PositiveInt = 8
    mlp_embedding_size: PositiveInt = 8
    mlp_layers: list[PositiveInt] = [16, 8, 4]
    
    loss: ImplicitLoss = "logistic"
    negative_strategy: NegativeStrategy | None = None
    negative_count: PositiveInt = 4
    positive_weight: float = 1.0
    
    user_bias: bool | None = False
    item_bias: bool = False
    convolution_layers: int = 0

    def selected_negative_strategy(self) -> NegativeStrategy:
        if self.negative_strategy is not None:
            return self.negative_strategy
        elif self.loss == "warp":
            return "misranked"
        else:
            return "uniform"


class FlexMFNCFModel(FlexMFModel):
    """
    Torch module for Neural Collaborative Filtering (NCF).
    """

    def __init__(self, gmf_e_size: int, mlp_e_size: int, mlp_layers: list[int],
                 n_users: int, n_items: int, rng: torch.Generator, init_scale: float = 0.1, sparse: bool = False):
        # We call the super class with layers=0 and without bias.
        # This gives us u_embed and i_embed which we use as GMF embeddings.
        super().__init__(
            e_size=gmf_e_size, n_users=n_users, n_items=n_items, rng=rng,
            user_bias=False, item_bias=False, init_scale=init_scale, sparse=sparse, layers=0
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
        
        # Final output layer
        self.prediction = nn.Linear(input_size + gmf_e_size, 1)
        nn.init.kaiming_uniform_(self.prediction.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user: Tensor, item: Tensor, *, return_norm: bool = False):
        u_gmf = self.u_embed(user)
        i_gmf = self.i_embed(item)
        
        u_mlp = self.u_mlp_embed(user)
        i_mlp = self.i_mlp_embed(item)
        
        # Ensure MLP inputs are broadcasted to the same shape before concatenating
        if u_mlp.shape != i_mlp.shape:
            u_mlp = u_mlp.expand(i_mlp.shape[:-1] + (u_mlp.shape[-1],))
            i_mlp = i_mlp.expand(u_mlp.shape[:-1] + (i_mlp.shape[-1],))
            
        gmf_out = u_gmf * i_gmf
        mlp_out = self.mlp(torch.cat([u_mlp, i_mlp], dim=-1))
        
        out = torch.cat([gmf_out, mlp_out], dim=-1)
        score = self.prediction(out).squeeze(-1)
        
        if return_norm:
            # Return regularizations
            l2 = norm(u_gmf, dim=-1) + norm(i_gmf, dim=-1) + norm(u_mlp, dim=-1) + norm(i_mlp, dim=-1)
            if l2.shape != score.shape:
                # Shape adjustment handled externally if broadcast differs
                pass
            return torch.stack((score, l2))
            
        return score


class FlexMFNCFTrainer(FlexMFImplicitTrainer):
    """
    Trainer for the NCF Model. Repurposes ImplicitTrainer's loop.
    """
    
    def create_model(self) -> FlexMFNCFModel:
        return FlexMFNCFModel(
            gmf_e_size=self.config.gmf_embedding_size,  # type: ignore
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
