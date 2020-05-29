"""
Utilities for parallel processing.
"""

import os
import multiprocessing as mp
from multiprocessing.queues import SimpleQueue
import functools as ft
import logging
import inspect
from concurrent.futures import ProcessPoolExecutor
from abc import ABC, abstractmethod
import pickle

from ..sharing import persist

if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle

_log = logging.getLogger(__name__)
__work_model = None
__work_func = None


def _p5_recv(self):
    buf = self.recv_bytes()
    return pickle.loads(buf)


def _p5_send(self, obj):
    buf = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    self._send_bytes(buf)


class FastQ(SimpleQueue):
    """
    SimpleQueue subclass that uses Pickle5 instead of default pickling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__patch()

    def __patch(self):
        # monkey-patch the sockets to use pickle5
        self._reader.recv = _p5_recv.__get__(self._reader)
        self._writer.send = _p5_send.__get__(self._writer)

    def get(self):
        with self._rlock:
            res = self._reader.recv_bytes()
        return pickle.loads(res)

    def put(self, obj):
        bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        # follow SimpleQueue, need to deal with _wlock being None
        if self._wlock is None:
            self._writer.send_bytes(bytes)
        else:
            with self._wlock:
                self._writer.send_bytes(bytes)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__patch()


class LKContext(mp.context.SpawnContext):
    def SimpleQueue(self):
        return FastQ(ctx=self.get_context())


LKContext.INSTANCE = LKContext()


def _initialize_worker(mkey, func):
    global __work_model, __work_func, __profile
    __work_model = mkey
    __work_func = func


def _proc_worker(*args):
    model = __work_model.get()
    return __work_func(model, *args)


def proc_count(core_div=2, max_default=None):
    """
    Get the number of desired jobs for multiprocessing operations.  This does not
    affect Numba or MKL multithreading.

    This count can come from a number of sources:
    * The ``LK_NUM_PROCS`` environment variable
    * The number of CPUs, divided by ``core_div`` (default 2)

    Args:
        core_div(int or None):
            The divisor to scale down the number of cores; ``None`` to turn off core-based
            fallback.
        max_default:
            The maximum number of processes to use if the environment variable is not
            configured.

    Returns:
        int: The number of jobs desired.
    """

    nprocs = os.environ.get('LK_NUM_PROCS')
    if nprocs is not None:
        nprocs = int(nprocs)
    elif core_div is not None:
        nprocs = max(mp.cpu_count() // core_div, 1)
        if max_default is not None:
            nprocs = min(nprocs, max_default)

    return nprocs


def invoker(model, func, n_jobs=None):
    """
    Get an appropriate invoker for performing oeprations on ``model``.

    Args:
        model(obj): The model object on which to perform operations.
        func(functio): The function to call.  The function must be pickleable.
        n_jobs(int or None):
            The number of processes to use for parallel operations.  If ``None``, will
            call :func:`proc_count` with a maximum default process count of 4.

    Returns:
        ModelOpInvoker:
            An invoker to perform operations on the model.
    """
    if n_jobs is None:
        n_jobs = proc_count(max_default=4)

    if n_jobs == 1:
        return InProcessOpInvoker(model, func)
    elif 'mp_context' in inspect.signature(ProcessPoolExecutor).parameters:
        return ProcessPoolOpInvoker(model, func, n_jobs)
    else:
        _log.warn('using multiprocessing.Pool, upgrade to Python 3.7 for best results')
        return MPOpInvoker(model, func, n_jobs)


class ModelOpInvoker(ABC):
    """
    Interface for invoking operations on a model, possibly in parallel.  The operation
    invoker is configured with a model and a function to apply, and applies that function
    to the arguments supplied in `map`.

    An invoker is a context manager that calls :meth:`shutdown` when exited.
    """

    @abstractmethod
    def map(self, *iterables):
        """
        Apply the configured function to the model and iterables.  This is like :py:func:`map`,
        except it supplies the invoker's model as the first object to ``func``.

        Args:
            iterables: Iterables of arguments to provide to the function.

        Returns:
            iterable: An iterable of the results.
        """
        pass

    def shutdown(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


class InProcessOpInvoker(ModelOpInvoker):
    def __init__(self, model, func):
        _log.info('setting up in-process worker')
        self.model = model
        self.function = func

    def map(self, *iterables):
        proc = ft.partial(self.function, self.model)
        return map(proc, *iterables)


class ProcessPoolOpInvoker(ModelOpInvoker):
    def __init__(self, model, func, n_jobs):
        key = persist(model)
        ctx = LKContext.INSTANCE
        _log.info('setting up ProcessPoolExecutor w/ %d workers', n_jobs)
        self.executor = ProcessPoolExecutor(n_jobs, ctx, _initialize_worker, (key, func))

    def map(self, *iterables):
        return self.executor.map(_proc_worker, *iterables)

    def shutdown(self):
        self.executor.shutdown()


class MPOpInvoker(ModelOpInvoker):
    def __init__(self, model, func, n_jobs):
        key = persist(model)
        ctx = LKContext.INSTANCE
        _log.info('setting up multiprocessing.Pool w/ %d workers', n_jobs)
        self.pool = ctx.Pool(n_jobs, _initialize_worker, (key, func))

    def map(self, *iterables):
        return self.pool.starmap(_proc_worker, zip(*iterables))

    def shutdown(self):
        self.pool.close()
