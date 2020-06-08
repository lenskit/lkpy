"""
Support for sharing and saving models and data structures.
"""

import os
import warnings
from abc import abstractmethod, ABC
from contextlib import contextmanager
import threading
import logging

_log = logging.getLogger(__name__)

_store_state = threading.local()


def _save_mode():
    return getattr(_store_state, 'mode', 'save')


@contextmanager
def sharing_mode():
    """
    Context manager to tell models that pickling will be used for cross-process
    sharing, not model persistence.
    """
    old = _save_mode()
    _store_state.mode = 'share'
    try:
        yield
    finally:
        _store_state.mode = old


def in_share_context():
    """
    Query whether sharing mode is active.  If ``True``, we are currently in a
    :func:`sharing_mode` context, which means model pickling will be used for
    cross-process sharing.
    """
    return _save_mode() == 'share'


class PersistedModel(ABC):
    """
    A persisted model for inter-process model sharing.

    These objects can be pickled for transmission to a worker process.

    .. note::
        Subclasses need to override the pickling protocol to implement the
        proper pickling implementation.
    """

    @abstractmethod
    def get(self):
        """
        Get the persisted model, reconstructing it if necessary.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Release the persisted model resources.  Should only be called in the
        parent process (will do nothing in a child process).
        """
        pass

    def transfer(self):
        """
        Mark an object for ownership transfer.  This object, when pickled, will
        unpickle into an owning model that frees resources when closed. Used to
        transfer ownership of shared memory resources from child processes to
        parent processes.  Such an object should only be unpickled once.

        The default implementation sets the ``is_owner`` attribute to ``'transfer'``.

        Returns:
            ``self`` (for convenience)
        """
        if not self.is_owner:
            warnings.warning('non-owning objects should not be transferred', stacklevel=1)
        else:
            self.is_owner = 'transfer'
        return self


def persist(model, *, method=None):
    """
    Persist a model for cross-process sharing.

    This will return a persiste dmodel that can be used to reconstruct the model
    in a worker process (using :func:`reconstruct`).

    If no method is provided, this function automatically selects a model persistence
    strategy from the the following, in order:

    1. If `LK_TEMP_DIR` is set, use :mod:`binpickle` in shareable mode to save
       the object into the LensKit temporary directory.
    2. If :mod:`multiprocessing.shared_memory` is available, use :mod:`pickle`
       to save the model, placing the buffers into shared memory blocks.
    3. Otherwise, use :mod:`binpickle` in shareable mode to save the object
       into the system temporary directory.

    Args:
        model(obj):
            The model to persist.
        method(str or None):
            The method to use.  Can be one of ``binpickle`` or ``shm``.

    Returns:
        PersistedModel: The persisted object.
    """
    if method is not None:
        if method == 'binpickle':
            method = persist_binpickle
        elif method == 'shm':
            method = persist_shm
        elif not hasattr(method, '__call__'):
            raise ValueError('invalid method %s: must be one of binpickle, shm, or a funciton')

    if method is None:
        if SHM_AVAILABLE and 'LK_TEMP_DIR' not in os.environ:
            method = persist_shm
        else:
            method = persist_binpickle

    return method(model)


from .binpickle import persist_binpickle, BPKPersisted     # noqa: E402,F401
from .shm import persist_shm, SHMPersisted, SHM_AVAILABLE  # noqa: E402,F401
