"""
Rust acceleration code.
"""

from __future__ import annotations

import pyarrow as pa

from lenskit.data.types import NPMatrix, NPVector
from lenskit.funksvd import FunkSVDTrainingData, FunkSVDTrainingParams

from . import als, data, slim

__all__ = [
    "als",
    "data",
    "slim",
    "init_accel_pool",
    "thread_count",
    "FunkSVDTrainer",
]

def init_accel_pool(n_threads: int): ...
def thread_count() -> int: ...

class FunkSVDTrainer:
    def __init__(
        self,
        config: FunkSVDTrainingParams,
        data: FunkSVDTrainingData,
        user_features: NPMatrix,
        item_features: NPMatrix,
    ): ...
    def feature_epoch(self, feature: int, estimates: NPVector, trail: float) -> float: ...

def sparse_row_debug_type(arr: pa.Array) -> tuple[str, int, int]: ...
def sparse_structure_debug_large(arr: pa.Array) -> tuple[int, int, int]: ...
