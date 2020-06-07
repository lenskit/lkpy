"""
Utilities for parallel processing.
"""

import os
import multiprocessing as mp
from multiprocessing.queues import SimpleQueue
from multiprocessing.connection import Listener, Client, wait
import threading
import functools as ft
import logging
import logging.handlers
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
__is_worker = False


def is_worker():
    return __is_worker


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


class InjectHandler:
    "Handler that re-injects a message into parent process logging"
    level = logging.DEBUG

    def handle(self, record):
        logger = logging.getLogger(record.name)
        logger.handle(record)


def _initialize_worker(mkey, func, threads, queue):
    global __work_model, __work_func, __is_worker
    __work_model = mkey
    __work_func = func
    __is_worker = True

    if queue is not None:
        h = logging.handlers.QueueHandler(queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        h.setLevel(logging.DEBUG)

    import numba
    numba.config.NUMBA_NUM_THREADS = threads
    try:
        import mkl
        _log.debug('configuring Numba thread count')
        mkl.set_num_threads(threads)
    except ImportError:
        pass

    _log.debug('worker %d ready', os.getpid())


def _proc_worker(*args):
    model = __work_model.get()
    return __work_func(model, *args)


def proc_count(core_div=2, max_default=None, level=0):
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
        level:
            The process nesting level.  0 is the outermost level of parallelism; subsequent
            levels control nesting.  Levels deeper than 1 are rare, and it isn't expected
            that callers actually have an accurate idea of the threading nesting, just that
            they are configuring a child.  If the process count is unconfigured, then level
            1 will use ``core_div``, and deeper levels will use 1.

    Returns:
        int: The number of jobs desired.
    """

    nprocs = os.environ.get('LK_NUM_PROCS', None)
    if nprocs is not None:
        nprocs = [int(s) for s in nprocs.split(',')]
    elif core_div is not None:
        nprocs = max(mp.cpu_count() // core_div, 1)
        if max_default is not None:
            nprocs = min(nprocs, max_default)
        nprocs = [nprocs, core_div]

    if level >= len(nprocs):
        return 1
    else:
        return nprocs[level]


def invoker(model, func, n_jobs=None):
    """
    Get an appropriate invoker for performing oeprations on ``model``.

    Args:
        model(obj): The model object on which to perform operations.
        func(function): The function to call.  The function must be pickleable.
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
    to the arguments supplied in `map`.  Child process invokers also route logging messages
    to the parent process, so logging works even with multiprocessing.

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
        kid_tc = proc_count(level=1)
        log_queue = ctx.Queue()
        self.log_listener = logging.handlers.QueueListener(log_queue, InjectHandler())
        self.log_listener.start()
        self.executor = ProcessPoolExecutor(n_jobs, ctx, _initialize_worker,
                                            (key, func, kid_tc, log_queue))

    def map(self, *iterables):
        return self.executor.map(_proc_worker, *iterables)

    def shutdown(self):
        self.executor.shutdown()
        self.log_listener.stop()


class MPOpInvoker(ModelOpInvoker):
    def __init__(self, model, func, n_jobs):
        key = persist(model)
        ctx = LKContext.INSTANCE
        kid_tc = proc_count(level=1)
        log_queue = ctx.Queue()
        self.log_listener = logging.handlers.QueueListener(log_queue, InjectHandler())
        self.log_listener.start()
        _log.info('setting up multiprocessing.Pool w/ %d workers', n_jobs)
        self.pool = ctx.Pool(n_jobs, _initialize_worker, (key, func, kid_tc, log_queue))

    def map(self, *iterables):
        return self.pool.starmap(_proc_worker, zip(*iterables))

    def shutdown(self):
        self.pool.close()
        self.log_listener.stop()
