"""
Rust acceleration code.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from lenskit.data.matrix import SparseRowArray
from lenskit.data.types import NPMatrix, NPVector
from lenskit.funksvd import FunkSVDTrainingData, FunkSVDTrainingParams

from . import als, data

__all__ = [
    "als",
    "data",
    "init_accel_pool",
    "thread_count",
    "NegativeSampler",
    "FunkSVDTrainer",
]

def init_accel_pool(n_threads: int): ...
def thread_count() -> int: ...
def sample_negatives(
    coords: data.CoordinateTable,
    rows: pa.Int32Array,
    n_cols: int,
    *,
    max_attempts: int = 10,
    pop_weighted: bool = False,
    seed: int,
): ...

class RowColumnSet:
    def __init__(self, matrix: SparseRowArray): ...
    def contains_pair(self, row: int, col: int) -> bool: ...

class NegativeSampler:
    def __init__(self, rc_set: RowColumnSet, users: pa.Int32Array, tgt_n: int): ...
    def num_remaining(self) -> int: ...
    def accumulate(self, items: np.ndarray[tuple[int], np.dtype[np.int32]], force: bool): ...
    def result(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]: ...

class FunkSVDTrainer:
    def __init__(
        self,
        config: FunkSVDTrainingParams,
        data: FunkSVDTrainingData,
        user_features: NPMatrix,
        item_features: NPMatrix,
    ): ...
    def feature_epoch(self, feature: int, estimates: NPVector, trail: float) -> float: ...
