"""
Rust acceleration code.
"""

from __future__ import annotations

from typing import TypedDict

import pyarrow as pa

from lenskit.data.types import NPMatrix, NPVector
from lenskit.funksvd import FunkSVDTrainingData, FunkSVDTrainingParams

from . import als, data, knn, slim

__all__ = [
    "als",
    "data",
    "knn",
    "slim",
    "init_accel_pool",
    "thread_count",
    "FunkSVDTrainer",
]

def init_accel_pool(n_threads: int): ...
def thread_count() -> int: ...

class _LogMsg(TypedDict):
    level: str
    logger: str
    message: str

class AccelLogListener:
    def __init__(self): ...
    def get_message(self) -> _LogMsg | None: ...
    def update_level(self, level: str): ...

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
