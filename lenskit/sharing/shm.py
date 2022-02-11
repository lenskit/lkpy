import sys
import logging
import pickle
from . import sharing_mode, PersistedModel

try:
    import multiprocessing.shared_memory as shm
    SHM_AVAILABLE = sys.platform != 'win32'
except ImportError:
    SHM_AVAILABLE = False

if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle

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

    with sharing_mode():
        data = pickle.dumps(model, protocol=5, buffer_callback=buffers.append)

    total_size = sum(memoryview(b).nbytes for b in buffers)
    _log.info('serialized %s to %d pickle bytes with %d buffers of %d bytes',
              model, len(data), len(buffers), total_size)

    if buffers:
        # blit the buffers to the SHM block
        _log.debug('preparing to share %d buffers', len(buffers))
        memory = shm.SharedMemory(create=True, size=total_size)
        cur_offset = 0
        blocks = []
        for i, buf in enumerate(buffers):
            ba = buf.raw()
            blen = ba.nbytes
            bend = cur_offset + blen
            _log.debug('saving %d bytes in buffer %d/%d', blen, i+1, len(buffers))
            memory.buf[cur_offset:bend] = ba
            blocks.append((cur_offset, bend))
            cur_offset = bend
    else:
        memory = None
        blocks = []

    return SHMPersisted(data, memory, blocks)


class SHMPersisted(PersistedModel):
    buffers = []
    _model = None
    memory = None

    def __init__(self, data, memory, blocks):
        self.pickle_data = data
        self.blocks = blocks
        self.memory = memory
        self.shm_name = memory.name if memory is not None else None
        self.is_owner = True

    def get(self):
        if self._model is None:
            _log.debug('loading model from shared memory')
            shm = self._open()
            buffers = []
            for bs, be in self.blocks:
                buffers.append(shm.buf[bs:be])

            self._model = pickle.loads(self.pickle_data, buffers=buffers)

        return self._model

    def close(self, unlink=True):
        self._model = None

        _log.debug('releasing SHM buffers')
        self.buffers = None
        if self.memory is not None:
            self.memory.close()
            if unlink and self.is_owner and self.is_owner != 'transfer':
                self.memory.unlink()
                self.is_owner = False
            self.memory = None

    def _open(self):
        if self.shm_name and not self.memory:
            self.memory = shm.SharedMemory(name=self.shm_name)
        return self.memory

    def __getstate__(self):
        return {
            'pickle_data': self.pickle_data,
            'blocks': self.blocks,
            'shm_name': self.shm_name,
            'is_owner': True if self.is_owner == 'transfer' else False
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.is_owner:
            _log.debug('opening shared buffers after ownership transfer')
            self._open()

    def __del__(self):
        self.close(False)
