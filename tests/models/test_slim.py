# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pyarrow as pa

from lenskit._accel import slim as _slim_accel
from lenskit.data import Dataset
from lenskit.data.matrix import SparseRowArray
from lenskit.knn.slim import SLIMConfig, SLIMScorer
from lenskit.logging import get_logger
from lenskit.parallel.config import ensure_parallel_init
from lenskit.testing import ScorerTests

_log = get_logger(__name__)


def test_slim_trainer(ml_ds: Dataset):
    "Test internal SLIM training function."
    ensure_parallel_init()
    ui_matrix = ml_ds.interactions().matrix().csr_structure(format="arrow")
    iu_matrix = ui_matrix.transpose()

    result = _slim_accel.train_slim(ui_matrix, iu_matrix, 0.005, 0.01, 10, None)
    result = pa.chunked_array(result).combine_chunks()
    result = SparseRowArray.from_array(result)
    assert isinstance(result, SparseRowArray)
    assert result.shape == (ml_ds.item_count, ml_ds.item_count)
    _log.info("received result", nnz=result.nnz)


class TestSLIM(ScorerTests):
    component = SLIMScorer
    config = SLIMConfig(max_iters=10)
    expected_ndcg = (0.01, 0.2)
