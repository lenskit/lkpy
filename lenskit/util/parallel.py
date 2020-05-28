"""
Utilities for parallel processing.
"""

import os
import multiprocessing as mp
import functools as ft
import logging
import inspect
from concurrent.futures import ProcessPoolExecutor
from abc import ABC, abstractmethod

from ..sharing import persist

_log = logging.getLogger(__name__)
__work_model = None
__work_func = None


def _initialize_worker(mkey, func):
    global __work_model, __work_func
    __work_model = mkey
    __work_func = func


def _proc_worker(*args):
    model = __work_model.get()
    return __work_func(model, *args)


def proc_count(core_div=2):
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

    Returns:
        int: The number of jobs desired.
    """

    nprocs = os.environ.get('LK_NUM_PROCS')
    if nprocs is not None:
        nprocs = int(nprocs)
    elif core_div is not None:
        nprocs = max(mp.cpu_count() // core_div, 1)

    return nprocs


def invoker(model, func, n_jobs=None):
    """
    Get an appropriate invoker for performing oeprations on ``model``.

    Args:
        model(obj): The model object on which to perform operations.
        func(functio): The function to call.  The function must be pickleable.
        n_jobs(int or None):
            The number of processes to use for parallel operations.  If ``None``, will
            call :func:`proc_count`.

    Returns:
        ModelOpInvoker:
            An invoker to perform operations on the model.
    """
    if n_jobs is None:
        n_jobs = proc_count()

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
        self.model = model
        self.function = func

    def map(self, *iterables):
        proc = ft.partial(self.function, self.model)
        return map(proc, *iterables)


class ProcessPoolOpInvoker(ModelOpInvoker):
    def __init__(self, model, func, n_jobs):
        key = persist(model)
        ctx = mp.get_context('spawn')
        self.executor = ProcessPoolExecutor(n_jobs, ctx, _initialize_worker, (key, func))

    def map(self, *iterables):
        return self.executor.map(_proc_worker, *iterables)

    def shutdown(self):
        self.executor.shutdown()


class MPOpInvoker(ModelOpInvoker):
    def __init__(self, model, func, n_jobs):
        key = persist(model)
        ctx = mp.get_context('spawn')
        self.pool = ctx.Pool(n_jobs, _initialize_worker, (key, func))

    def map(self, *iterables):
        return self.pool.starmap(_proc_worker, zip(*iterables))

    def shutdown(self):
        self.pool.close()
