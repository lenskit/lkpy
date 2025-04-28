# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Support code for managing model state.

Many LensKit components are built on machine learning models of various forms.
While :mod:`lenskit.training` provides support for training those models,
this module provides support for managing their learned state and weights.
"""

from ._container import ParameterContainer

__all__ = [
    "ParameterContainer",
]
