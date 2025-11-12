# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest import fixture, skip

from lenskit import batch
from lenskit.knn.item import ItemKNNScorer
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import TTSplit
from lenskit.testing import msweb


def test_knn_batch_msweb(msweb: TTSplit):
    pipe = topn_pipeline(ItemKNNScorer(feedback="implicit"))
    pipe.train(msweb.train)

    recs = batch.recommend(pipe, msweb.test, 10)
    assert recs.key_fields == ("session_id",)
    assert len(recs) == len(msweb.test)
