# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field, replace
from typing import Self

import numpy as np
import structlog
import torch
from torch import Tensor

from lenskit.data import MatrixRelationshipSet
from lenskit.logging import get_logger

_log = get_logger(__name__)


@dataclass
class FlexMFTrainingContext:
    """
    Context information for training the FlexMF models.

    Stability:
        Experimental
    """

    device: str
    """
    Torch device for training.
    """

    rng: np.random.Generator
    """
    NumPy generator for random number generation.
    """

    torch_rng: torch.Generator
    """
    PyTorch RNG for initialization and generation.
    """

    log: structlog.stdlib.BoundLogger = field(default_factory=lambda: _log.bind())
    """
    A logger, that is bound the current training status / position.
    """


@dataclass
class FlexMFTrainingData:
    """
    Training data for a FlexMF model.

    Stability:
        Experimental
    """

    batch_size: int

    n_users: int
    n_items: int

    users: torch.Tensor | np.ndarray[tuple[int], np.dtype[np.int32]]
    "User numbers for training samples."
    items: torch.Tensor | np.ndarray[tuple[int], np.dtype[np.int32]]
    "Item numbers for training samples."

    matrix: MatrixRelationshipSet | None = None
    """
    The original relationship set we are training on.
    """

    fields: dict[str, torch.Tensor] = field(default_factory=dict)
    "Additional per-sample data fields."

    def to(self, device: str) -> Self:
        """
        Move this data to another device.
        """
        ut = torch.as_tensor(self.users, device=device)
        it = torch.as_tensor(self.items, device=device)
        fts = {f: torch.as_tensor(t, device=device) for (f, t) in self.fields.items()}
        return replace(self, users=ut, items=it, fields=fts)

    @property
    def n_samples(self) -> int:
        return len(self.users)

    def epoch(self, context: FlexMFTrainingContext) -> FlexMFTrainingEpoch:
        # permute the data
        perm = np.require(context.rng.permutation(self.n_samples), dtype=np.int32)
        # convert to tensor, send to the training data's device.
        if isinstance(self.items, torch.Tensor):
            perm = torch.tensor(perm).to(self.items.device)

        return FlexMFTrainingEpoch(self, perm)


@dataclass
class FlexMFTrainingEpoch:
    """
    Permuted data for a single training epoch.
    """

    data: FlexMFTrainingData
    permutation: torch.Tensor | np.ndarray[tuple[int], np.dtype[np.int32]]

    @property
    def n_samples(self) -> int:
        return self.data.n_samples

    @property
    def batch_size(self) -> int:
        return self.data.batch_size

    @property
    def batch_count(self) -> int:
        return math.ceil(self.n_samples / self.batch_size)

    def batches(
        self, fields: Sequence[str] | set[str] | None = None
    ) -> Generator[FlexMFTrainingBatch, None, None]:
        """
        Iterate over batches in this epoch.

        Args:
            fields:
                The fields to include in the batch.
        """

        fields = set(self.data.fields.keys()) if fields is None else fields
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            rows = self.permutation[start:end]

            yield self.make_batch(rows, fields)

    def make_batch(
        self,
        rows: Tensor | np.ndarray[tuple[int], np.dtype[np.int32]],
        fields: Sequence[str] | set[str],
    ) -> FlexMFTrainingBatch:
        ut = self.data.users[rows]
        it = self.data.items[rows]
        fts = {f: self.data.fields[f][rows] for f in fields}
        return FlexMFTrainingBatch(self.data, ut, it, fts)


@dataclass
class FlexMFTrainingBatch:
    "Representation of a single batch."

    data: FlexMFTrainingData
    users: torch.Tensor | np.ndarray[tuple[int], np.dtype[np.int32]]
    items: torch.Tensor | np.ndarray[tuple[int], np.dtype[np.int32]]
    fields: dict[str, torch.Tensor]

    def to(self, device: str) -> Self:
        """
        Move this data to another device.
        """
        ut = torch.as_tensor(self.users, device=device)
        it = torch.as_tensor(self.items, device=device)
        fts = {f: torch.as_tensor(t, device=device) for (f, t) in self.fields.items()}
        return replace(self, users=ut, items=it, fields=fts)
