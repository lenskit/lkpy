# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
PyTorch utility functions.
"""

import functools

import torch


def inference_mode(func):
    """
    Function decorator that puts PyTorch in inference mode.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return func(*args, **kwargs)

    return wrapper
