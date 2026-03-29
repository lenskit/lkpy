from collections.abc import Sequence

from lenskit.data.matrix import SparseRowArray
from lenskit.logging import Progress

def train_slim(
    ui_matrix: SparseRowArray,
    iu_matrix: SparseRowArray,
    l1_reg: float,
    l2_reg: float,
    max_iters: int,
    progress: Progress | None,
) -> Sequence[SparseRowArray]: ...
