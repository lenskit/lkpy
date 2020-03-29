import os
import logging
import tempfile
from pathlib import Path

import joblib
from ..util import scratch_dir
from . import BaseModelStore, BaseModelClient, sharing_mode

_log = logging.getLogger(__name__)


class FileClient(BaseModelClient):
    """
    Client using Joblib's memory-mapping pickle support.
    """

    _last_key = None

    def get_model(self, key):
        if self._last_key == key:
            _log.debug('reusing model %s', key)
        else:
            _log.debug('loading model from %s', key)
            self._last_model = joblib.load(key, mmap_mode='r')
        return self._last_model

    def __getstate__(self):
        if isinstance(self, BaseModelStore):
            raise RuntimeError('stores cannot be pickled')
        else:
            return {}  # nothing to pickle here


class FileModelStore(BaseModelStore, FileClient):
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
            self._path = Path(tempfile.mkdtemp(prefix='lk-share-', dir=scratch_dir()))
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

    def client(self):
        return FileClient()

    def put_model(self, model):
        fd, fn = tempfile.mkstemp('.model', 'lk-joblib', self._path)
        fpath = Path(fn)
        _log.debug('saving model %s to %s', model, fpath)
        os.close(fd)

        with sharing_mode():
            joblib.dump(model, fn)
        self._files.append(fpath)
        return fpath

    def put_serialized(self, path):
        if self.reserialize:
            return super().put_serialized(path)
        else:
            return path

    def __str__(self):
        if self.path is not None:
            return f'FileModelStore({self.path})'
        else:
            return 'FileModelStore()'
