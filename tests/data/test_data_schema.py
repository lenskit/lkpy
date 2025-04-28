# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.data.schema import CURRENT_VERSION, OLDEST_VERSION, DataSchema


def test_schema_default_version():
    schema = DataSchema(name="foo")
    assert schema.name == "foo"
    assert schema.version == CURRENT_VERSION


def test_schema_load_compat_version():
    schema = DataSchema.model_validate_json("{}")
    assert schema.version == OLDEST_VERSION
