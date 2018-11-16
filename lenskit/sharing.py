"""
Support for sharing data between processes to enable multi-process
evaluation operations more easily.
"""

import os
from abc import ABCMeta, abstractmethod, abstractclassmethod
import tempfile
import subprocess
from pathlib import Path
import uuid
import logging
import weakref

import numpy as np
import pandas as pd

import pyarrow as pa
try:
    from pyarrow import plasma
    have_plasma = True
except ImportError:
    have_plasma = False

_logger = logging.getLogger(__name__)

_context_stack = []


def current_context():
    if _context_stack:
        return _context_stack[-1]


def _push_context(ctx):
    _context_stack.append(ctx)


def _pop_context():
    _context_stack.pop()


class ShareContext(metaclass=ABCMeta):
    """
    Base class for sharing contexts.
    """

    @abstractmethod
    def put_array(self, array):
        pass

    @abstractmethod
    def get_array(self, key):
        pass

    def put_series(self, series):
        i_k = self.put_array(series.index.values)
        v_k = self.put_array(series.values)
        return (series.index.name, i_k, series.name, v_k)

    def get_series(self, key):
        i_n, i_k, s_n, v_k = key
        index = self.get_array(i_k)
        values = self.get_array(v_k)
        series = pd.Series(values, index=index)
        series.name = s_n
        series.index.name = i_n
        return series


class DiskShareContext(ShareContext):
    def __init__(self, path=None):
        if path is None:
            self._tmp_dir = tempfile.TemporaryDirectory(prefix='lkpy')
            self.path = Path(self._tmp_dir.name)
        else:
            self.path = Path(path)
            self._tmp_dir = None

    def put_array(self, array):
        key = uuid.uuid4()
        fn = self.path / key.hex
        fn = fn.with_suffix('.npy')
        np.save(fn, array)
        return key

    def get_array(self, key):
        fn = self.path / key.hex
        fn = fn.with_suffix('.npy')
        return np.load(fn)

    def __getstate__(self):
        return self.path

    def __setstate__(self, state):
        self.path = state
        self._tmp_dir = None

    def __enter__(self):
        _push_context(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._tmp_dir is not None:
                self._tmp_dir.cleanup()
        finally:
            _pop_context()


class PlasmaShareContext(ShareContext):
    def __init__(self, path=None, size=None, _child=False):
        if path is None:
            path = os.environ.get('PLASMA_SOCKET', None)
        if size is None:
            size = os.environ.get('PLASMA_SIZE', 4*1024*1024*1024)
            size = str(size)

        if path is None:
            self._dir = tempfile.TemporaryDirectory(prefix='lkpy-plasma')
            self.socket = Path(self._dir.name) / 'plasma-socket'
            _logger.info('launching Plasma store with %s bytes', size)
            self.process = subprocess.Popen(['plasma_store', '-m', size, '-s', self.socket],
                                            stdin=subprocess.DEVNULL)
        else:
            self.socket = path
            self.process = None

        self.client = plasma.connect(str(self.socket), "", 0)
        self.__obj_map = {}

    def _rand_id(self):
        return plasma.ObjectID(np.random.bytes(20))
    
    def _clear_ref(self, id):
        try:
            del self.__obj_map[id]
        except KeyError:
            pass

    def put_array(self, array):
        key = self._rand_id()
        _logger.debug('storing array of shape %s in %s', array.shape, key)
        tensor = pa.Tensor.from_numpy(array)
        size = pa.get_tensor_size(tensor)
        _logger.debug('array data is %d bytes', size)
        buf = self.client.create(key, size)

        sink = pa.FixedSizeBufferWriter(buf)
        pa.write_tensor(tensor, sink)

        self.client.seal(key)

        return key

    def get_array(self, key):
        [data] = self.client.get_buffers([key])
        _logger.debug('loading data of size %d from %s', data.size, key)
        buffer = pa.BufferReader(data)
        tensor = pa.read_tensor(buffer)
        result = tensor.to_numpy()
        rid = id(result)
        
        self.__obj_map[rid] = data
        weakref.finalize(result, self._clear_ref, rid)
        
        _logger.debug('loaded array of shape %s', result.shape)
        return result

    def __getstate__(self):
        return self.path

    def __setstate__(self, state):
        self.path = state
        self.process = None
        self.client = plasma.connect(self.path, "", 0)

    def __enter__(self):
        _push_context(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._cleanup()
        finally:
            _pop_context()

    def __del__(self):
        self._cleanup()

    def _cleanup(self):
        if self.client is not None:
            _logger.debug('disconnecting from Plasma')
            self.client.disconnect()
            self.client = None

        if self.process is not None:
            _logger.info('shutting down Plasma')
            self.process.terminate()
            self.process.wait()
            self.process = None
            self._dir.cleanup()


class Shareable(metaclass=ABCMeta):
    @abstractmethod
    def share_publish(self, context):
        raise NotImplemented()

    @abstractclassmethod
    def share_resolve(self, key, context):
        raise NotImplemented()


share_impls = [DiskShareContext]
if have_plasma:
    share_impls.append(PlasmaShareContext)
