# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
PyTorch utility functions.
"""

import functools

import numpy as np
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


def safe_tensor(array) -> torch.Tensor:
    """
    Safely convert an array into a NumPy tensor.  This includes copying it to
    writable memory if necessary.
    """
    if torch.is_tensor(array):
        return array

    arr = np.asarray(array)
    arr = np.require(arr, requirements="W")
    return torch.from_numpy(arr)
