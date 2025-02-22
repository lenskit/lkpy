# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Serialization utilities for parallel processing.
"""

import io
import logging
import pickle
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Any, NamedTuple

import torch
from torch.multiprocessing.reductions import reduce_storage, reduce_tensor

_log = logging.getLogger(__name__)


class SHMData(NamedTuple):
    """
    Serialized data (with shared memory handles).
    """

    pickle: bytes
    buffers: list[tuple[SharedMemory | None, int]]


class SHMPickler(pickle.Pickler):
    manager: SharedMemoryManager | None
    buffers: list[tuple[SharedMemory | None, int]]

    def __init__(
        self,
        file,
        protocol: int | None = pickle.HIGHEST_PROTOCOL,
        manager: SharedMemoryManager | None = None,
        *,
        fix_imports: bool = False,
    ) -> None:
        super().__init__(file, protocol, fix_imports=fix_imports, buffer_callback=self._buffer_cb)
        self.manager = manager
        self.buffers = []

    def _buffer_cb(self, buffer: pickle.PickleBuffer):
        mem = buffer.raw()
        if mem.nbytes == 0:
            shm = None
        elif self.manager:
            shm = self.manager.SharedMemory(mem.nbytes)
        else:
            shm = SharedMemory(create=True, size=mem.nbytes)

        # copy the data
        if shm is not None:
            shm.buf[: mem.nbytes] = mem
        self.buffers.append((shm, mem.nbytes))

    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            if obj.is_sparse_csr:
                return torch.sparse_csr_tensor, (
                    obj.crow_indices(),
                    obj.col_indices(),
                    obj.values(),
                    obj.shape,
                )
            elif obj.is_sparse:
                return torch.sparse_coo_tensor, (
                    obj.indices(),
                    obj.values(),
                    obj.shape,
                )
            elif obj.layout == torch.sparse_csc:
                return torch.sparse_csc_tensor, (
                    obj.ccol_indices(),
                    obj.row_indices(),
                    obj.values(),
                    obj.shape,
                )
            else:
                return reduce_tensor(obj)

        if isinstance(obj, torch.UntypedStorage):
            return reduce_storage(obj)

        return NotImplemented


def shm_serialize(obj: Any, manager: SharedMemoryManager | None = None) -> SHMData:
    """
    Serialize an object for processing in a subclass with shared memory when
    feasible (including CUDA).
    """
    out = io.BytesIO()
    pkl = SHMPickler(out, pickle.HIGHEST_PROTOCOL, manager)
    pkl.dump(obj)

    data = out.getvalue()
    _log.debug("serialized model into %s pickle bytes", len(data))
    return SHMData(bytes(data), pkl.buffers)


def shm_deserialize(data: SHMData) -> Any:
    """
    Deserialize SHM-pickled data.
    """
    buffers = [(shm.buf[:n] if shm is not None else b"") for shm, n in data.buffers]
    return pickle.loads(data.pickle, buffers=buffers)
