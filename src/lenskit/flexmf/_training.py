from __future__ import annotations

import math
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field, replace
from typing import Self

import numpy as np
import torch
from torch import Tensor


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

    users: torch.Tensor
    "User numbers for training samples."
    items: torch.Tensor
    "Item numbers for training samples."
    fields: dict[str, torch.Tensor] = field(default_factory=dict)
    "Additional per-sample data fields."

    def to(self, device: str) -> Self:
        """
        Move this data to another device.
        """
        ut = self.users.to(device)
        it = self.items.to(device)
        fts = {f: t.to(device) for (f, t) in self.fields.items()}
        return replace(self, user_nums=ut, item_nums=it, fields=fts)

    @property
    def n_samples(self) -> int:
        return len(self.users)

    def epoch(self, context: FlexMFTrainingContext) -> FlexMFTrainingEpoch:
        # permute the data
        perm = context.rng.permutation(self.n_samples)
        # convert to tensor, send to the training data's device.
        perm = torch.tensor(perm).to(self.items.device)

        return FlexMFTrainingEpoch(self, perm)


@dataclass
class FlexMFTrainingEpoch:
    """
    Permuted data for a single training epoch.
    """

    data: FlexMFTrainingData
    permutation: torch.Tensor

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

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)

            rows = self.permutation[start:end]

            fields = set(self.data.fields.keys()) if fields is None else fields
            yield self.make_batch(rows, fields)

    def make_batch(self, rows: Tensor, fields: Sequence[str] | set[str]) -> FlexMFTrainingBatch:
        ut = self.data.users[rows]
        it = self.data.items[rows]
        fts = {f: self.data.fields[f][rows] for f in fields}
        return FlexMFTrainingBatch(ut, it, fts)


@dataclass
class FlexMFTrainingBatch:
    "Representation of a single batch."

    users: torch.Tensor
    items: torch.Tensor
    fields: dict[str, torch.Tensor]

    def to(self, device: str) -> Self:
        """
        Move this data to another device.
        """
        ut = self.users.to(device)
        it = self.items.to(device)
        fts = {f: t.to(device) for (f, t) in self.fields.items()}
        return replace(self, users=ut, items=it, fields=fts)
