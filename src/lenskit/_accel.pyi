from __future__ import annotations

from typing import Protocol

import numpy as np
import pyarrow as pa

from lenskit.data.matrix import SparseRowArray
from lenskit.data.types import NPMatrix

class _ALSAccelerator(Protocol):
    @staticmethod
    def train_explicit_matrix_cd(
        matrix: SparseRowArray,
        this: NPMatrix,
        other: NPMatrix,
        reg: float,
    ) -> float: ...

als: _ALSAccelerator

class RowColumnSet:
    def __init__(self, matrix: SparseRowArray): ...
    def contains_pair(self, row: int, col: int) -> bool: ...

class NegativeSampler:
    def __init__(self, rc_set: RowColumnSet, users: pa.Int32Array, tgt_n: int): ...
    def num_remaining(self) -> int: ...
    def accumulate(self, items: np.ndarray[tuple[int], np.dtype[np.int32]], force: bool): ...
    def result(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]: ...
