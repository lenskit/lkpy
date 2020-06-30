import os
import pathlib
import tempfile
import logging
import gc

import binpickle

from . import sharing_mode, PersistedModel

_log = logging.getLogger(__name__)


def persist_binpickle(model, dir=None, file=None):
    """
    Persist a model using binpickle.

    Args:
        model: The model to persist.
        dir: The temporary directory for persisting the model object.
        file: The file in which to save the object.

    Returns:
        PersistedModel: The persisted object.
    """
    if file is not None:
        path = pathlib.Path(file)
    else:
        if dir is None:
            dir = os.environ.get('LK_TEMP_DIR', None)
        fd, path = tempfile.mkstemp(suffix='.bpk', prefix='lkpy-', dir=dir)
        os.close(fd)
        path = pathlib.Path(path)
    _log.debug('persisting %s to %s', model, path)
    with binpickle.BinPickler.mappable(path) as bp, sharing_mode():
        bp.dump(model)
    return BPKPersisted(path)


class BPKPersisted(PersistedModel):
    def __init__(self, path):
        self.path = path
        self.is_owner = True
        self._bpk_file = None
        self._model = None

    def get(self):
        if self._bpk_file is None:
            _log.debug('loading %s', self.path)
            self._bpk_file = binpickle.BinPickleFile(self.path, direct=True)
            self._model = self._bpk_file.load()
        return self._model

    def close(self, unlink=True):
        if self._bpk_file is not None:
            self._model = None
            try:
                _log.debug('closing BPK file')
                try:
                    self._bpk_file.close()
                except BufferError:
                    _log.debug('could not close %s, collecting garbage and retrying', self.path)
                    gc.collect()
                    self._bpk_file.close()
            except (BufferError, IOError) as e:
                _log.warn('error closing %s: %s', self.path, e)
            self._bpk_file = None

        if self.is_owner and unlink:
            assert self._model is None
            if unlink:
                _log.debug('deleting %s', self.path)
                try:
                    self.path.unlink()
                except IOError as e:
                    _log.warn('could not remove %s: %s', self.path, e)
            self.is_owner = False

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_bpk_file'] = None
        d['_model'] = None
        if self.is_owner == 'transfer':
            d['is_owner'] = True
        else:
            d['is_owner'] = False
        return d

    def __del___(self):
        self.close(False)
