# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Serialization utilities for parallel processing.
"""

import logging
import pickle
from multiprocessing.reduction import ForkingPickler
from typing import Any

import numpy as np
import torch
from torch.multiprocessing.reductions import reduce_tensor

_log = logging.getLogger(__name__)


def _rebuild_ndarray(tensor):
    return tensor.numpy()


def _reduce_ndarray(a: np.ndarray):
    try:
        t = torch.from_numpy(a)
        return (_rebuild_ndarray, (t,))
    except TypeError:
        return a.__reduce__()


def _reduce_tensor_wrapper(t: torch.Tensor):
    if t.is_sparse_csr:
        return torch.sparse_csr_tensor, (
            t.crow_indices(),
            t.col_indices(),
            t.values(),
            t.shape,
        )
    elif t.is_sparse:
        return torch.sparse_coo_tensor, (
            t.row_indices(),
            t.col_indices(),
            t.values(),
            t.shape,
        )
    else:
        return reduce_tensor(t)


def init_reductions():
    ForkingPickler.register(np.ndarray, _reduce_ndarray)
    ForkingPickler.register(torch.Tensor, _reduce_tensor_wrapper)


def shm_serialize(obj: Any) -> bytes:
    """
    Serialize an object for processing in a subclass with shared memory when
    feasible (including CUDA).
    """
    data = ForkingPickler.dumps(obj, pickle.HIGHEST_PROTOCOL)
    _log.debug("serialized model into %s pickle bytes", len(data))
    return bytes(data)


def shm_deserialize(data) -> Any:
    """
    Deserialize SHM-pickled data.
    """
    return pickle.loads(data)
