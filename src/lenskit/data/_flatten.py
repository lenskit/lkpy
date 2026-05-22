# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Utility functions to flatten and unflatten dictionaries.
"""

from typing import Any


def flatten_dict(data: dict[str, Any]) -> dict[str, Any]:
    out = {}
    _insert_flattened(out, data)
    return out


def _insert_flattened(dst: dict[str, Any], data: dict[str, Any], prefix: str = ""):
    for key, value in data.items():
        fk = prefix + key
        if isinstance(value, dict):
            _insert_flattened(dst, value, fk + ".")
        else:
            dst[fk] = value


def unflatten_dict(data: dict[str, Any], *, sep=".") -> dict[str, Any]:
    out = {}
    for key, value in data.items():
        parts = key.split(sep)
        tgt = out
        for k in parts[:-1]:
            tgt = tgt.setdefault(k, {})
        tgt[parts[-1]] = value
    return out
