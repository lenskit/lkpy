from __future__ import annotations

from typing import Protocol

import numpy as np
import pyarrow as pa

from lenskit.data.matrix import SparseRowArray
from lenskit.data.types import NPMatrix, NPVector
from lenskit.funksvd import FunkSVDTrainingData, FunkSVDTrainingParams
from lenskit.logging import Progress

def init_accel_pool(n_threads: int): ...

class RowColumnSet:
    def __init__(self, matrix: SparseRowArray): ...
    def contains_pair(self, row: int, col: int) -> bool: ...

class NegativeSampler:
    def __init__(self, rc_set: RowColumnSet, users: pa.Int32Array, tgt_n: int): ...
    def num_remaining(self) -> int: ...
    def accumulate(self, items: np.ndarray[tuple[int], np.dtype[np.int32]], force: bool): ...
    def result(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]: ...

class _DataAccelerator(Protocol):
    @staticmethod
    def is_sorted_coo(data: list[pa.RecordBatch], c1: str, c2: str) -> bool: ...

data: _DataAccelerator

class _ALSAccelerator(Protocol):
    @staticmethod
    def train_explicit_matrix(
        matrix: SparseRowArray,
        this: NPMatrix,
        other: NPMatrix,
        reg: float,
        pb: Progress | None,
    ) -> float: ...
    @staticmethod
    def train_implicit_matrix(
        matrix: SparseRowArray,
        this: NPMatrix,
        other: NPMatrix,
        otor: NPMatrix,
        pb: Progress | None,
    ) -> float: ...

als: _ALSAccelerator

class FunkSVDTrainer:
    def __init__(
        self,
        config: FunkSVDTrainingParams,
        data: FunkSVDTrainingData,
        user_features: NPMatrix,
        item_features: NPMatrix,
    ): ...
    def feature_epoch(self, feature: int, estimates: NPVector, trail: float) -> float: ...
