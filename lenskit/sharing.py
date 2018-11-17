"""
Support for sharing data between processes to enable multi-process
evaluation operations more easily.
"""

from abc import ABCMeta, abstractmethod, abstractclassmethod
import tempfile
from pathlib import Path
import uuid
import logging
from multiprocessing import sharedctypes as mpctypes

import numpy as np
import pandas as pd

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

    def child(self):
        return DiskShareContext(self.path)

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


class SHMShareContext(ShareContext):
    def __init__(self):
        pass

    def put_array(self, array):
        _logger.debug('sharing object of type %s and shape %s (size=%d)',
                      array.dtype.str, array.shape, array.size)
        code = np.ctypeslib._typecodes[array.dtype.str]
        shared = mpctypes.Array(code, array.size, lock=False)
        shape = array.shape
        nda = np.ctypeslib.as_array(shared)
        nda[:] = array.reshape(array.size)

        return (shape, shared)

    def get_array(self, key):
        shape, shared = key
        nda = np.ctypeslib.as_array(shared)
        nda = nda.reshape(shape)
        return nda

    def child(self):
        return self

    def __enter__(self):
        _push_context(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _pop_context()


class Shareable(metaclass=ABCMeta):
    @abstractmethod
    def share_publish(self, context):
        raise NotImplementedError()

    @abstractclassmethod
    def share_resolve(self, key, context):
        raise NotImplementedError()


share_impls = [SHMShareContext, DiskShareContext]


def context():
    return SHMShareContext()
