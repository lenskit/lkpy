"""
Support for sharing and saving models and data structures.
"""

import sys
import warnings
from abc import abstractmethod
from contextlib import contextmanager
import threading
import logging
import pickle

_log = logging.getLogger(__name__)

_store_state = threading.local()


def _save_mode():
    return getattr(_store_state, 'mode', 'save')


def _active_stores():
    if not hasattr(_store_state, 'active'):
        _store_state.active = []
    return _store_state.active


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


def get_store(reuse=True, *, in_process=False):
    """
    Get a model store, using the best available on the current platform.  The
    resulting store should be used as a context manager, as in:

    >>> with get_store() as store:
    ...     pass

    This function uses the following priority list for locating a suitable store:

    1. The currently-active store, if ``reuse=True``
    2. A no-op store, if ``in_process=True``
    3. :class:`SHMModelStore`, if on Python 3.8
    4. :class:`JoblibModelStore`

    Args:
        reuse(bool):
            If a store is active (with a ``with`` block), use that store instead
            of creating a new one.
        in_process(bool):
            If ``True``, then create a no-op store for use without multiprocessing.

    Returns:
        BaseModelStore: the model store.
    """
    stores = _active_stores()
    if reuse and stores:
        return stores[-1]
    elif in_process:
        return NoopModelStore()
    elif SHMModelStore.ENABLED:
        return SHMModelStore()
    else:
        return FileModelStore()


class BaseModelClient:
    """
    Model store client to get models given keys.  Clients must be able to be cheaply
    pickled and de-pickled to enable worker processes to access them.
    """

    @abstractmethod
    def get_model(self, key):
        """
        Get a model from the  model store.

        Args:
            key: the model key to retrieve.

        Returns:
            SharedObject:
                The model, previously stored with :meth:`BaseModelStore.put_model`,
                wrapped in a :class:`SharedObject` to manage underlying resources.
        """


class SharedObject:
    """
    Wrapper for a shared object that can release it when the object is no longer needed.

    Objects of this type are context managers, that return the *shared object* (not
    themselves) when entered.

    Any other refernces to ``object``, or its contents, **must** be released before
    calling :meth:`release` or exiting the context manager.  Among other things, that
    means that you will need to delete its variable::

        with client.get_model(k) as model:
            # model here is the actual model object wrapped by the SharedObject
            # returned by get_model
            pass       # actually do the things you want to do
            del model  # release model, so the shared object can be closed

    Be careful of stray references to the model object!  Some things we have seen
    causing stray references include:

    * passing the algorithm to a logger (call :func:`str` on it explicitly), at least
      in the test harness

    The default implementation uses :func:`sys.getrefcount` to provide debugging
    support to help catch stray references.

    Attributes:
        object: the underlying shared object.
    """
    _rc = None
    _exiting = False

    def __init__(self, obj):
        self.object = obj

    def release(self):
        """
        Release the shared object.  Automatically called by :meth:`__exit__`, so in
        normal use of a shared object with a ``with`` statement, this method is not
        needed.

        The base class implementation simply deletes the object reference.  Subclasses
        should override this method to handle their own release logic.
        """
        if self.object is not None:
            rc = sys.getrefcount(self.object)
            if self._rc and rc > self._rc:
                # since most backends won't actually crash, track & emit a warning
                _log.debug('reference count to %s increased from %d to %d',
                           self.object, self._rc, rc)
                wm = f'reference count to {self.object} increased, object leak?'
                level = 3 if self._exiting else 2
                warnings.warn(wm, ResourceWarning, stacklevel=level)
            del self.object

    def __enter__(self):
        self._rc = sys.getrefcount(self.object)
        return self.object

    def __exit__(self, *args):
        self._exiting = True
        try:
            self.release()
        finally:
            del self._exiting


class BaseModelStore(BaseModelClient):
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
            a key to retrieve the model with :meth:`BaseModelClient.get_model`
        """
        pass

    def put_serialized(self, path):
        """
        Deserialize a model and load it into the store.

        The base class method unpickles ``path`` and calls :meth:`put_model`.
        """
        with open(path, 'rb') as mf:
            return self.put_model(pickle.load(mf))

    @abstractmethod
    def client(self):
        """
        Get a client for the model store.  Clients are cheap to pass to
        child processes for multiprocessing.

        Returns:
            BaseModelClient: the model client.
        """
        pass

    def init(self):
        "Initialize the store."

    def shutdown(self):
        "Shut down the store"

    def __enter__(self):
        if self._act_count == 0:
            self.init()
        self._act_count = self._act_count + 1
        _active_stores().append(self)
        return self

    def __exit__(self, *args):
        self._act_count = self._act_count - 1
        if self._act_count == 0:
            self.shutdown()
        assert _active_stores()[-1] is self
        _active_stores().pop()
        return None

    def __getstate__(self):
        raise RuntimeError('stores cannot be pickled, do you want to use the client?')


class NoopModelStore(BaseModelStore):
    """
    Model store that does nothing - models are their own keys.  Only useful in
    single-threaded computations.
    """

    def get_model(self, key):
        return SharedObject(key)

    def put_model(self, model):
        return model

    def client(self):
        return self  # since we're only single-threaded, we are the client

    def __str__(self):
        return 'NoopModelStore'


# more imports
from .file import FileModelStore
from .sharedmem import SHMModelStore
