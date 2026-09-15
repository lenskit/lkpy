# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Schemas for LensKit configuration and specification files.

This package collects the schemas used by various configuration capabilities
in one subpackage, so they are easy to find and can reference each other
without as many import problems as we have when they live inside the packages
for the different subsystems they control.
"""

from . import settings
from ._load import load_model_data

__all__ = [
    "load_model_data",
    "settings",
]
