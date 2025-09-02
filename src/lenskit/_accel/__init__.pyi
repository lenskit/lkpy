"""
Rust acceleration code.
"""

from __future__ import annotations

import numpy as np

from lenskit.data.types import NPMatrix, NPVector
from lenskit.funksvd import FunkSVDTrainingData, FunkSVDTrainingParams

from . import als, data

__all__ = [
    "als",
    "data",
    "init_accel_pool",
    "thread_count",
    "FunkSVDTrainer",
]

def init_accel_pool(n_threads: int): ...
def thread_count() -> int: ...
def sample_negatives(
    coords: data.CoordinateTable,
    rows: np.ndarray[tuple[int], np.dtype[np.int32]],
    n_cols: int,
    *,
    max_attempts: int = 10,
    pop_weighted: bool = False,
    seed: int,
) -> np.ndarray[tuple[int, int], np.dtype[np.int32]]: ...

class FunkSVDTrainer:
    def __init__(
        self,
        config: FunkSVDTrainingParams,
        data: FunkSVDTrainingData,
        user_features: NPMatrix,
        item_features: NPMatrix,
    ): ...
    def feature_epoch(self, feature: int, estimates: NPVector, trail: float) -> float: ...
