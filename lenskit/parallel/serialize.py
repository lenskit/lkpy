# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Serialization utilities for parallel processing.
"""

import pickle
from multiprocessing.reduction import ForkingPickler
from typing import Any

import numpy as np
import torch


def _rebuild_ndarray(tensor):
    return tensor.numpy()


def _reduce_ndarray(a: np.ndarray):
    print("serializing", a)
    print(f"{a.dtype} {a.shape}")
    try:
        t = torch.from_numpy(a)
        return (_rebuild_ndarray, (t,))
    except TypeError:
        return a.__reduce__()


def init_reductions():
    ForkingPickler.register(np.ndarray, _reduce_ndarray)


def shm_serialize(obj: Any) -> bytes:
    """
    Serialize an object for processing in a subclass with shared memory when
    feasible (including CUDA).
    """
    return ForkingPickler.dumps(obj, pickle.HIGHEST_PROTOCOL)


def shm_deserialize(data) -> Any:
    """
    Deserialize pickled data.
    """
    return pickle.loads(data)
