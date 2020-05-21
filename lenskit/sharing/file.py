import os
import logging
import tempfile
from pathlib import Path

import binpickle
from ..util import scratch_dir
from . import BaseModelStore, BaseModelClient, SharedObject, sharing_mode

_log = logging.getLogger(__name__)


class BPKObject(SharedObject):
    """
    BinPickle-specific shared object class that manages the file reference.
    """

    def __init__(self, file, object):
        super(BPKObject, self).__init__(object)
        self.count = 1
        self.bp_file = file

    def incr(self):
        self.count += 1
        return self

    def release(self):
        self.count -= 1
        if self.count <= 0:
            _log.debug('releasing %s', str(self.object))
            del self.object
            self.bp_file.close()
            del self.bp_file

    def __del__(self):
        # enforce a deletion order
        del self.object
        del self.bp_file


class FileClient(BaseModelClient):
    """
    Client using BinPickle's memory-mapping pickle support.
    """

    _last_key = None
    _last_model = None

    def get_model(self, key):
        if self._last_key == key:
            _log.debug('reusing model %s', key)
        else:
            if self._last_model:
                self._last_model.release()
                self._last_model = None

            _log.debug('loading model from %s', key)
            bpf = binpickle.BinPickleFile(key, direct=True)
            obj = bpf.load()
            self._last_model = BPKObject(bpf, obj)
        return self._last_model.incr()

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
        fd, fn = tempfile.mkstemp('.model', 'lk-bpk', self._path)
        fpath = Path(fn)
        _log.debug('saving model %s to %s', model, fpath)
        os.close(fd)

        with sharing_mode():
            binpickle.dump(model, fn, mappable=True)

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
