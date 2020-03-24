"""
Sharing support using Python 3.8 shared memory.
"""

from typing import NamedTuple
import logging
import multiprocessing.shared_memory as shm
import pickle
import uuid

import numpy as np

from . import BaseModelClient, BaseModelStore, sharing_mode

_log = logging.getLogger(__name__)


class SHMKey(NamedTuple):
    id: uuid.UUID
    data: bytes
    buffers: list

    def __str__(self):
        nbs = len(self.data)
        return f'SHMKey({self.id}: {nbs} bytes)'


class SHMClient(BaseModelClient):
    _last_key = None
    _last_model = None
    _last_bufs = None

    def get_model(self, key: SHMKey):
        if self._last_key and self._last_key.id == key.id:
            _log.debug('reusing model %s', key)
        else:
            self._last_model = None
            self._last_bufs = None

            _log.debug('loading model from %s', key)
            buffers = []
            shm_bufs = []
            for bn, bs in key.buffers:
                # funny business with buffer sizes
                block = shm.SharedMemory(name=bn)
                _log.debug('%s: %d bytes (%d used)', block.name, bs, block.size)
                buffers.append(block.buf[:bs])
                shm_bufs.append(block)
            self._last_model = pickle.loads(key.data, buffers=buffers)
            self._last_bufs = shm_bufs
            self._last_key = key

        return self._last_model

    def __getstate__(self):
        if isinstance(self, BaseModelStore):
            raise RuntimeError('stores cannot be pickled')
        else:
            return {}  # nothing to pickle here


class SHMModelStore(BaseModelStore, SHMClient):
    """
    Model store using shared memory and Pickle Protocol 5.

    Args:
        path:
            the path to use; otherwise uses a new temp directory under
            :func:`util.scratch_dir`.
        reserialize:
            if ``True`` (the default), models passed to :meth:`put_serialized` are
            re-serialized in the SHM storage.
    """

    def init(self):
        self.buffers = {}

    def shutdown(self, *args):
        for k, bs in self.buffers.items():
            for buf in bs:
                buf.close()
                buf.unlink()
        del self.buffers

    def client(self):
        return SHMClient()

    def put_model(self, model):
        buffers = []
        buf_keys = []

        def buf_cb(buf):
            ba = buf.raw()
            block = shm.SharedMemory(create=True, size=ba.nbytes)
            _log.debug('serializing %d bytes to %s', ba.nbytes, block.name)
            block.buf[:] = ba
            buffers.append(block)
            buf_keys.append((block.name, ba.nbytes))

        id = uuid.uuid4()

        with sharing_mode():
            data = pickle.dumps(model, protocol=5, buffer_callback=buf_cb)
            shm_bytes = sum(b.size for b in buffers)
            _log.info('serialized %s to %s (%d pickle bytes and %d buffers of %d bytes)',
                      model, id, len(data), len(buffers), shm_bytes)

        self.buffers[id] = buffers

        return SHMKey(id, data, buf_keys)

    def __str__(self):
        return 'SHMModelStore()'
