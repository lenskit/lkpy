"""
Support for sharing and saving models and data structures.
"""

from abc import abstractmethod
from contextlib import contextmanager, AbstractContextManager
import logging
import pickle

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


def get_store(reuse=True, *, in_process=False):
    """
    Get a model store, using the best available on the current platform.  The
    resulting store should be used as a context manager, as in:

    >>> with get_store() as store:
    >>>     pass

    .. note:: This is not thread-safe.

    Args:
        reuse(bool):
            If a store is active (with a ``with`` block), use that store instead
            of creating a new one.
        in_process(bool):
            If ``True``, then create a no-op store for use without multiprocessing.

    Returns:
        BaseModelStore: the model store.
    """
    if reuse and _active_stores:
        return _active_stores[-1]
    elif in_process:
        return NoopModelStore()
    elif 'SHMModelStore' in globals():
        return SHMModelStore()
    else:
        return JoblibModelStore()


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
            The model, previously stored with :meth:`put_model`.
        """


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
            a key to retrieve the model with :meth:`get_model`
        """
        pass

    def put_serialized(self, path):
        """
        Deserialize a model and load it into the store.

        The base class method unpickles ``path`` and calls :meth:`store`.
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
        _active_stores.append(self)
        return self

    def __exit__(self, *args):
        self._act_count = self._act_count - 1
        if self._act_count == 0:
            self.shutdown()
        assert _active_stores[-1] is self
        _active_stores.pop()
        return None

    def __getstate__(self):
        raise RuntimeError('stores cannot be pickled, do you want to use the client?')


class NoopModelStore(BaseModelStore):
    """
    Model store that does nothing - models are their own keys.  Only useful in
    single-threaded computations.
    """

    def get_model(self, key):
        return key

    def put_model(self, model):
        return model

    def client(self):
        return self  # since we're only single-threaded, we are the client

    def __str__(self):
        return 'NoopModelStore'


## more imports
from .joblib import JoblibModelStore
try:
    from .sharedmem import SHMModelStore
except ImportError:
    pass  # shared memory not available
