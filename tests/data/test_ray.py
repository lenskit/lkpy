# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest import importorskip, mark, skip

from lenskit.data import Dataset, load_movielens
from lenskit.parallel.ray import ensure_cluster

ray = importorskip("ray")

data_dir = Path("data")


@mark.slow
@mark.parametrize("name", ["ml-latest-small", "ml-100k.zip", "ml-20m.zip"])
def test_ray_roundtrip(name):
    ensure_cluster()
    ds_path = data_dir / name
    if not ds_path.exists():
        skip(f"dataset {name} not available")

    ds = load_movielens(ds_path)

    ref = ray.put(ds)
    ds2 = ray.get(ref)

    assert isinstance(ds2, Dataset)
    assert ds2 is not ds

    assert ds2.item_count == ds.item_count
