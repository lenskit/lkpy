import logging
import pickle
try:
    import multiprocessing.shared_memory as shm
    SHM_AVAILABLE = True
except ImportError:
    SHM_AVAILABLE = False

if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle

from . import sharing_mode, PersistedModel

_log = logging.getLogger(__name__)


def persist_shm(model, dir=None):
    """
    Persist a model using binpickle.

    Args:
        model: The model to persist.
        dir: The temporary directory for persisting the model object.

    Returns:
        PersistedModel: The persisted object.
    """
    if shm is None:
        raise ImportError('multiprocessing.shared_memory')

    buffers = []
    buf_keys = []

    def buf_cb(buf):
        ba = buf.raw()
        block = shm.SharedMemory(create=True, size=ba.nbytes)
        _log.debug('serializing %d bytes to %s', ba.nbytes, block.name)
        # blit the buffer into shared memory
        block.buf[:ba.nbytes] = ba
        buffers.append(block)
        buf_keys.append((block.name, ba.nbytes))

    with sharing_mode():
        data = pickle.dumps(model, protocol=5, buffer_callback=buf_cb)
        shm_bytes = sum(b.size for b in buffers)
        _log.info('serialized %s to %d pickle bytes with %d buffers of %d bytes',
                  model, len(data), len(buffers), shm_bytes)

    return SHMPersisted(data, buf_keys, buffers)


class SHMPersisted(PersistedModel):
    buffers = []
    _model = None

    def __init__(self, data, buf_specs, buffers):
        self.pickle_data = data
        self.buffer_specs = buf_specs
        self.buffers = buffers
        self.is_owner = True

    def get(self):
        if self._model is None:
            _log.debug('loading model from shared memory')
            shm_bufs = self._open_buffers()
            buffers = []
            for (bn, bs), block in zip(self.buffer_specs, shm_bufs):
                # funny business with buffer sizes
                _log.debug('%s: %d bytes (%d used)', block.name, bs, block.size)
                buffers.append(block.buf[:bs])

            self._model = pickle.loads(self.pickle_data, buffers=buffers)

        return self._model

    def close(self, unlink=True):
        self._model = None
        if self.is_owner:
            _log.debug('releasing SHM buffers')
            for buf in self.buffers:
                buf.close()
                buf.unlink()
            del self.buffers
            self.is_owner = False

    def _open_buffers(self):
        if not self.buffers:
            self.buffers = [shm.SharedMemory(name=bn) for (bn, bs) in self.buffer_specs]
        return self.buffers

    def __getstate__(self):
        return {
            'pickle_data': self.pickle_data,
            'buffer_specs': self.buffer_specs,
            'is_owner': True if self.is_owner == 'transfer' else False
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.is_owner:
            _log.debug('opening shared buffers after ownership transfer')
            self._open_buffers
