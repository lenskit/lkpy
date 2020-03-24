"""
Support for sharing and saving models and data structures.
"""

import os
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager, AbstractContextManager
import logging
import pickle
import tempfile
from pathlib import Path

import joblib

from .util import scratch_dir

_log = logging.getLogger(__name__)

__save_mode = 'save'
_active_stores = []


@contextmanager
def sharing_mode():
    """
    Context manager to tell models that pickling will be used for cross-process
    sharing, not model persistence.
    """
    global __save_mode
    old = __save_mode
    __save_mode = 'share'
    try:
        yield
    finally:
        __save_mode = old


def in_share_context():
    """
    Query whether sharing mode is active.  If ``True``, we are currently in a
    :fun:`sharing_mode` context, which means model pickling will be used for
    cross-process sharing.
    """
    return __save_mode == 'share'


def get_store(reuse=True):
    """
    Get a model store, using the best available on the current platform.  The
    resulting store should be used as a context manager, as in:

    >>> with get_store() as store:
    >>>     pass

    Args:
        reuse(bool):
            If a store is active (with a ``with`` block), use that store instead
            of creating a new one.
    """
    if reuse and _active_stores:
        return _active_stores[-1]
    else:
        return JoblibModelStore()


class BaseModelStore(AbstractContextManager):
    """
    Base class for storing models for access across processes.

    Stores are also context managers that initalize themselves and clean themselves
    up.  As context managers, they are also re-entrant, and register themselves so
    that :func:`create_store` can re-use existing managers.
    """

    _act_count = 0

    @abstractmethod
    def put_model(self, model):
        """
        Store a model in the model store.

        Args:
            model(object): the model to store.

        Returns:
            a key to retrieve the model with :meth:`get_model`
        """
        pass

    @abstractmethod
    def get_model(self, key):
        """
        Get a model from the  model store.

        Args:
            key: the model key to retrieve.

        Returns:
            The model, previously stored with :meth:`put_model`.
        """

    def put_serialized(self, path):
        """
        Deserialize a model and load it into the store.

        The base class method unpickles ``path`` and calls :meth:`store`.
        """
        with open(path, 'rb') as mf:
            return self.put_model(pickle.load(mf))

    def init(self):
        "Initialize the store."

    def shutdown(self):
        "Shut down the store"

    def __enter__(self):
        if self._act_count == 0:
            self.init()
        self._act_count = self._act_count + 1
        _active_stores.append(self)
        return self

    def __exit__(self, *args):
        self._act_count = self._act_count - 1
        if self._act_count == 0:
            self.shutdown()
        assert _active_stores[-1] is self
        _active_stores.pop()
        return None


class JoblibModelStore(BaseModelStore):
    """
    Model store using JobLib's memory-mapping pickle support.

    Args:
        path:
            the path to use; otherwise uses a new temp directory under
            :func:`util.scratch_dir`.
        reserialize:
            if ``True`` (the default), models passed to :meth:`put_serialized` are
            re-serialized in the JobLib storage.
    """

    path: Path
    reserialize: bool

    def __init__(self, *, path=None, reserialize=True):
        self.path = path
        self.reserialize = reserialize

    def init(self):
        if self.path is None:
            self._path = Path(tempfile.mkdtemp(prefix='lk-share', dir=scratch_dir()))
            self._rmdir = True
        else:
            self._path = self.path
            self._rmdir = False

        self._files = []

    def shutdown(self, *args):
        failed = 0
        # delete files
        for f in self._files:
            try:
                _log.debug('removing %s', f)
                f.unlink()
            except PermissionError:
                failed += 1
                _log.warning('could not unlink %s', f)

        # clean up directory
        if failed:
            _log.warning('failed to delete %d temporary files from %s', failed, self.path)
        elif self._rmdir:
            try:
                self._path.rmdir()
            except IOError as e:
                _log.warning('could not delete %s: %s', self._path, e)

        # and clean up internal data structures
        del self._files
        del self._path
        del self._rmdir

    def put_model(self, model):
        fd, fn = tempfile.mkstemp('.model', 'lk-joblib', self._path)
        fpath = Path(fn)
        os.close(fd)

        with sharing_mode():
            joblib.dump(model, fn)
        self._files.append(fpath)
        return fpath

    def get_model(self, key):
        model = joblib.load(key, mmap_mode='r')
        return model

    def put_serialized(self, path):
        if self.reserialize:
            return super().put_serialized(path)
        else:
            return path
