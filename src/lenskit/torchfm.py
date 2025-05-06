# This file is part of LensKit.
# Copyright (c) 2019 rixwew
#	Copyright (C) 2018-2023 Boise State University
#	Copyright (C) 2023-2025 Drexel University
#	Licensed under the MIT license, see LICENSE.md for details.
#	SPDX-License-Identifier: MIT

import sys, os
if sys.path and os.path.basename(sys.path[0]) == "lenskit":
    sys.path.pop(0)
import logging
import numpy as np
import torch
from pydantic import BaseModel
from typing_extensions import override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_logger = logging.getLogger(__name__)

class TorchFMConfig(BaseModel):
    """Configuration for the LensKit wrapper around TorchFM."""
    embed_dim: int = 10
    epochs: int = 5
    lr: float = 1e-2
    batch_size: int = 1024
    negative_count: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.register_buffer(
            "offsets",
            torch.tensor((0, *np.cumsum(field_dims)[:-1]), dtype=torch.long),
        )
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        x = x + self.offsets.unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias

class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.register_buffer(
            "offsets",
            torch.tensor((0, *np.cumsum(field_dims)[:-1]), dtype=torch.long),
        )
        torch.nn.init.xavier_uniform_(self.embedding.weight)
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        x = x + self.offsets.unsqueeze(0)
        return self.embedding(x)

class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        sq = torch.sum(x, dim=1) ** 2
        ss = torch.sum(x * x, dim=1)
        ix = sq - ss
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FactorizationMachineModel(torch.nn.Module):
    """
    A PyTorch implementation of a Factorization Machine.
    """
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear    = FeaturesLinear(field_dims)
        self.fm        = FactorizationMachine(reduce_sum=True)
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        lin = self.linear(x)
        fm_ = self.fm(self.embedding(x))
        out = lin + fm_
        return torch.sigmoid(out.squeeze(1))

class TorchFMScorer(Component[ItemList], Trainable):
    """LensKit scorer wrapping the TorchFM Factorization Machine model."""
    config: TorchFMConfig

    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        # skip retraining if model already trained and retrain=False
        if not options.retrain and hasattr(self, 'model_'):
            return self
        # Load interactions
        # Load interactions
        matrix = data.interactions().matrix()
        coo    = matrix.coo_structure()
        users  = coo.row_numbers
        items  = coo.col_numbers

        # Save vocabs & build model
        self.users_ = data.users
        self.items_ = data.items
        dims = [data.user_count, data.item_count]

        model     = FactorizationMachineModel(dims, self.config.embed_dim).to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        loss_fn   = torch.nn.BCELoss()

        # Training loop
        for _ in range(self.config.epochs):
            # Uniform random negatives (guaranteed shape)
            neg = np.random.randint(
                0,
                data.item_count,
                size=(len(users), self.config.negative_count),
                dtype=np.int32,
            )

            pos_x = np.stack([users, items], axis=1)
            pos_y = np.ones(len(users), dtype=np.float32)

            # flatten neg per user
            neg_x = np.stack([
                np.repeat(users, self.config.negative_count),
                neg.flatten(),
            ], axis=1)
            neg_y = np.zeros(len(neg_x), dtype=np.float32)

            # merge pos + neg
            X = np.vstack([pos_x, neg_x])
            Y = np.concatenate([pos_y, neg_y])
            perm = np.random.permutation(len(Y))
            X, Y = X[perm], Y[perm]

            # minibatch SGD
            for start in range(0, len(Y), self.config.batch_size):
                end = start + self.config.batch_size
                bx  = torch.LongTensor(X[start:end]).to(self.config.device)
                by  = torch.FloatTensor(Y[start:end]).to(self.config.device)
                optimizer.zero_grad()
                preds = model(bx)
                loss  = loss_fn(preds, by)
                loss.backward()
                optimizer.step()

        self.model_ = model
        return self

    @override
    def __call__(self,
                 query: QueryInput,
                 items: ItemList) -> ItemList:
        q   = RecQuery.create(query)
        uid = self.users_.number(q.user_id, missing=None) if q.user_id else None
        if uid is None:
            return ItemList(items, scores=np.full(len(items), np.nan))

        inums = items.numbers(vocabulary=self.items_, missing="negative")
        mask  = inums >= 0
        valid = inums[mask]

        uarr = np.full_like(valid, uid)
        x    = np.stack([uarr, valid], axis=1)
        xb   = torch.LongTensor(x).to(self.config.device)

        with torch.no_grad():
            preds = self.model_(xb).cpu().numpy()

        scores       = np.full(len(items), np.nan, dtype=float)
        scores[mask] = preds
        return ItemList(items, scores=scores)
