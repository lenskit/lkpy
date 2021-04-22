"""
Utilities for parallel processing.
"""

import os
import multiprocessing as mp
from multiprocessing.queues import SimpleQueue
import functools as ft
import logging
import logging.handlers
import faulthandler
from concurrent.futures import ProcessPoolExecutor
from abc import ABC, abstractmethod
import pickle

from lenskit.sharing import persist, PersistedModel
from lenskit.util.log import log_queue
from lenskit.util.random import derive_seed, init_rng, get_root_seed

if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle

_log = logging.getLogger(__name__)
__work_model = None
__work_func = None
__is_worker = False


def is_worker():
    "Query whether the process is a worker, either for MP or for isolation."
    return __is_worker


def is_mp_worker():
    "Query whether the current process is a multiprocessing worker."
    return os.environ.get('_LK_IN_MP', 'no') == 'yes'


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


def _initialize_worker(log_queue, seed):
    "Initialize a worker process."
    global __is_worker
    __is_worker = True
    faulthandler.enable()
    if seed is not None:
        init_rng(seed)
    if log_queue is not None:
        h = logging.handlers.QueueHandler(log_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        h.setLevel(logging.DEBUG)


def _initialize_mp_worker(mkey, func, threads, log_queue, seed):
    seed = derive_seed(mp.current_process().name, base=seed)
    _initialize_worker(log_queue, seed)
    global __work_model, __work_func

    nnt_env = os.environ.get('NUMBA_NUM_THREADS', None)
    if nnt_env is None or int(nnt_env) > threads:
        _log.debug('configuring Numba thread count')
        import numba
        numba.config.NUMBA_NUM_THREADS = threads
    try:
        import mkl
        _log.debug('configuring MKL thread count')
        mkl.set_num_threads(threads)
    except ImportError:
        pass

    __work_model = mkey
    # deferred function unpickling to minimize imports before initialization
    __work_func = pickle.loads(func)

    _log.debug('worker %d ready (process %s)', os.getpid(), mp.current_process())


def _mp_invoke_worker(*args):
    model = __work_model.get()
    return __work_func(model, *args)


def _sp_worker(log_queue, seed, res_queue, func, args, kwargs):
    _initialize_worker(log_queue, seed)
    _log.debug('running %s in worker', func)
    try:
        res = func(*args, **kwargs)
        _log.debug('completed successfully')
        res_queue.put((True, res))
    except Exception as e:
        _log.error('failed, transmitting error %r', e)
        res_queue.put((False, e))


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


def run_sp(func, *args, **kwargs):
    """
    Run a function in a subprocess and return its value.  This is for achieving subprocess
    isolation, not parallelism.  The subprocess is configured so things like logging work
    correctly, and is initialized with a derived random seed.
    """
    ctx = LKContext.INSTANCE
    rq = ctx.SimpleQueue()
    seed = derive_seed()
    worker_args = (log_queue(), seed, rq, func, args, kwargs)
    _log.debug('spawning subprocess to run %s', func)
    proc = ctx.Process(target=_sp_worker, args=worker_args)
    proc.start()
    _log.debug('waiting for process %s to return', proc)
    success, payload = rq.get()
    _log.debug('received success=%s', success)
    _log.debug('waiting for process %s to exit', proc)
    proc.join()
    if proc.exitcode:
        _log.error('subprocess failed with code %d', proc.exitcode)
        raise RuntimeError('subprocess failed with code ' + str(proc.exitcode))
    if success:
        return payload
    else:
        _log.error('subprocess raised exception: %s', payload)
        raise ChildProcessError('error in child process', payload)


def invoker(model, func, n_jobs=None, *, persist_method=None):
    """
    Get an appropriate invoker for performing oeprations on ``model``.

    Args:
        model(obj): The model object on which to perform operations.
        func(function): The function to call.  The function must be pickleable.
        n_jobs(int or None):
            The number of processes to use for parallel operations.  If ``None``, will
            call :func:`proc_count` with a maximum default process count of 4.
        persist_method(str or None):
            The persistence method to use.  Passed as ``method`` to
            :func:`lenskit.sharing.persist`.

    Returns:
        ModelOpInvoker:
            An invoker to perform operations on the model.
    """
    if n_jobs is None:
        n_jobs = proc_count(max_default=4)

    if n_jobs == 1:
        return InProcessOpInvoker(model, func)
    else:
        return ProcessPoolOpInvoker(model, func, n_jobs, persist_method)


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
        if isinstance(model, PersistedModel):
            self.model = model.get()
        else:
            self.model = model
        self.function = func

    def map(self, *iterables):
        proc = ft.partial(self.function, self.model)
        return map(proc, *iterables)

    def shutdown(self):
        self.model = None


class ProcessPoolOpInvoker(ModelOpInvoker):
    _close_key = None

    def __init__(self, model, func, n_jobs, persist_method):
        if isinstance(model, PersistedModel):
            _log.debug('model already persisted')
            key = model
        else:
            _log.debug('persisting model with method %s', persist_method)
            key = persist(model, method=persist_method)
            self._close_key = key

        _log.debug('persisting function')
        func = pickle.dumps(func)
        ctx = LKContext.INSTANCE
        _log.info('setting up ProcessPoolExecutor w/ %d workers', n_jobs)
        os.environ['_LK_IN_MP'] = 'yes'
        kid_tc = proc_count(level=1)
        self.executor = ProcessPoolExecutor(n_jobs, ctx, _initialize_mp_worker,
                                            (key, func, kid_tc, log_queue(), get_root_seed()))

    def map(self, *iterables):
        return self.executor.map(_mp_invoke_worker, *iterables)

    def shutdown(self):
        self.executor.shutdown()
        os.environ.pop('_LK_IN_MP', 'yes')
        if self._close_key is not None:
            self._close_key.close()
            del self._close_key
