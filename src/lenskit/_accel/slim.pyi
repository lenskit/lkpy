from collections.abc import Sequence

from lenskit.data.matrix import SparseRowArray
from lenskit.parallel import AccelTask

def train_slim(
    ui_matrix: SparseRowArray,
    iu_matrix: SparseRowArray,
    l1_reg: float,
    l2_reg: float,
    max_iters: int,
    max_nbrs: int | None,
) -> AccelTask[Sequence[SparseRowArray]]: ...
