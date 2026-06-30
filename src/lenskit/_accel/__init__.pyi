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

class AtomicInt:
    """
    An atomic integer.

    .. note::

        All operations are done with the **relaxed** load state, so this class
        is only useful for counting, but not for synchronization.
    """
    def __init__(self, *, initial: int = 0): ...
    def load(self) -> int: ...
    def store(self, x: int) -> None: ...
    def fetch_add(self, incr: int = 1) -> int: ...

class NestedAccelPool:
    """
    Nested accelerator pools.  Don't use this directly, use
    :class:`lenskit.parallel.NestedPool` instead.
    """
    def __init__(self, n_threads: int): ...
    def shutdown(self) -> None: ...
