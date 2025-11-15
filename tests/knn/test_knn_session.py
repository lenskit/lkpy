# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest import fixture, skip

from lenskit import batch
from lenskit.data import RecQuery
from lenskit.knn.item import ItemKNNScorer
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import TTSplit
from lenskit.testing import msweb


def test_knn_batch_msweb(msweb: TTSplit):
    pipe = topn_pipeline(ItemKNNScorer(feedback="implicit"))
    pipe.train(msweb.train)

    queries = [
        RecQuery(query_id=k.session_id, session_items=il[:5])
        for (k, il) in msweb.test.items()
        if len(il) > 5
    ]
    _test = {k: il[5:] for (k, il) in msweb.test.items() if len(il) > 5}

    recs = batch.recommend(pipe, queries, 10)
    assert recs.key_fields == ("query_id",)
    assert len(recs) == len(queries)
