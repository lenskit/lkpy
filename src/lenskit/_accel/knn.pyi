from collections.abc import Sequence

import pyarrow as pa

from lenskit.data.matrix import SparseRowArray
from lenskit.logging import Progress

def compute_similarities(
    ui_ratings: SparseRowArray,
    iu_ratings: SparseRowArray,
    shape: tuple[int, int],
    min_sim: float,
    save_nbrs: int | None,
    progress: Progress | None,
) -> Sequence[SparseRowArray]: ...
def score_explicit(
    sims: SparseRowArray,
    ref_items: pa.Int32Array,
    ref_rates: pa.FloatArray,
    tgt_items: pa.Int32Array,
    max_nbrs: int,
    min_nbrs: int,
) -> pa.FloatArray: ...
def score_implicit(
    sims: SparseRowArray,
    ref_items: pa.Int32Array,
    tgt_items: pa.Int32Array,
    max_nbrs: int,
    min_nbrs: int,
) -> pa.FloatArray: ...
