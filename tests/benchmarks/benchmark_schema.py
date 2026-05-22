# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pyarrow as pa

from pytest import fixture, mark, skip


@mark.benchmark
def test_upgrade_null(benchmark):
    base = pa.schema({"item_id": pa.null()})
    new = pa.schema({"item_id": pa.int64(), "scores": pa.float64()})

    def upgrade():
        _up = pa.unify_schemas([base, new])

    benchmark(upgrade)


@mark.benchmark
def test_upgrade_self(benchmark):
    base = pa.schema({"item_id": pa.null()})
    new = pa.schema({"item_id": pa.int64(), "scores": pa.float64()})
    base = pa.unify_schemas([base, new])

    def upgrade():
        _up = pa.unify_schemas([base, new])

    benchmark(upgrade)


@mark.benchmark
def test_check_upgrade(benchmark):
    base = pa.schema({"item_id": pa.null()})
    new = pa.schema({"item_id": pa.int64(), "scores": pa.float64()})

    def upgrade():
        if not new.equals(base):
            _up = pa.unify_schemas([base, new])

    benchmark(upgrade)


@mark.benchmark
def test_check_noop(benchmark):
    base = pa.schema({"item_id": pa.null()})
    new = pa.schema({"item_id": pa.int64(), "scores": pa.float64()})
    base = pa.unify_schemas([base, new])

    def upgrade():
        if not new.equals(base):
            _up = pa.unify_schemas([base, new])

    benchmark(upgrade)


@mark.benchmark
def test_pycheck_noop(benchmark):
    base = pa.schema({"item_id": pa.null(), "rating": pa.float32()})
    new = pa.schema({"item_id": pa.int64(), "scores": pa.float64()})
    # base = pa.unify_schemas([base, new])

    def upgrade():
        good = True
        for i, n in enumerate(new.names):
            ty = new.types[i]
            f = base.field(n)
            if f.type != ty:
                good = False
                break

        if not good:
            _up = pa.unify_schemas([base, new])

    benchmark(upgrade)


@mark.benchmark
def test_create(benchmark):
    base = pa.schema({"item_id": pa.null()})
    new = pa.schema({"item_id": pa.int64(), "scores": pa.float64()})
    base = pa.unify_schemas([base, new])

    def upgrade():
        _new = pa.schema({"item_id": pa.int64(), "scores": pa.float64()})

    benchmark(upgrade)
