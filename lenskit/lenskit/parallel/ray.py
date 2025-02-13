"""
Support for parallelism with Ray.
"""

from __future__ import annotations

import base64
import itertools
import os
import pickle
from collections.abc import Callable, Iterable, Iterator
from platform import python_version
from typing import Generic

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from lenskit.logging import Task, get_logger
from lenskit.logging.worker import WorkerContext, WorkerLogConfig

from .config import (
    ParallelConfig,
    effective_cpu_count,
    get_parallel_config,
    initialize,
    subprocess_config,
)
from .invoker import A, InvokeOp, M, ModelOpInvoker, R

if python_version() < "3.12":
    RAY_SUPPORTED = False
else:
    RAY_SUPPORTED = RAY_AVAILABLE


LK_PROCESS_SLOT = "lk_process"
_worker_parallel: ParallelConfig
_worker_log: WorkerContext
_log = get_logger(__name__)


def ensure_cluster():
    if not ray.is_initialized():
        init_cluster()


def init_cluster(
    *,
    num_cpus: int | None = None,
    proc_slots: int | None = None,
    resources: dict[str, float] | None = None,
    worker_parallel: ParallelConfig | None = None,
    **kwargs,
):
    """
    Initialize or connect to a Ray cluster, with the LensKit options.

    Args:
        num_cpus:
            The total number of CPUs to allow. Defaults to
            :fun:`effective_cpu_count`.
        proc_slots:
            The number of “process slots” for LensKit parallel operations.
            Defaults to the LensKit process count.  These slots are recorded as
            the ``lk_process`` resource on the Ray cluster.
        resources:
            Additional custom resources to register in the Ray cluster.
        worker_parallel:
            Parallel processing configuration for worker processes.  If
            ``None``, uses the default.
        kwargs:
            Other options to pass to :func:`ray.init`.

    Stability:
        Experimental
    """
    global _worker_parallel
    if resources is None:
        resources = {}
    else:
        resources = resources.copy()

    cfg = get_parallel_config()
    if proc_slots is None:
        proc_slots = cfg.processes
    resources = {LK_PROCESS_SLOT: proc_slots}
    if num_cpus is None:
        num_cpus = effective_cpu_count()

    if worker_parallel is None:
        worker_parallel = subprocess_config()
    _worker_parallel = worker_parallel

    env = worker_parallel.env_vars().copy()
    wc = WorkerLogConfig.current()
    env["LK_LOG_CONFIG"] = base64.encodebytes(pickle.dumps(wc)).decode()

    runtime = ray.runtime_env.RuntimeEnv(env_vars=env, worker_process_setup_hook=_setup_ray_worker)

    _log.info("starting Ray cluster")
    ray.init(num_cpus=num_cpus, resources=resources, runtime_env=runtime, **kwargs)


def _setup_ray_worker():
    global _worker_log
    initialize()
    log_cfg = pickle.loads(base64.decodebytes(os.environb[b"LK_LOG_CONFIG"]))

    _worker_log = WorkerContext(log_cfg)
    _worker_log.start()


def inference_worker_cpus() -> int:
    return _worker_parallel.backend_threads


def training_worker_cpus() -> int:
    return _worker_parallel.total_threads


class RayOpInvoker(ModelOpInvoker[A, R], Generic[M, A, R]):
    function: InvokeOp[M, A, R]
    model_ref: ray.ObjectID

    def __init__(
        self,
        model: M,
        func: InvokeOp[M, A, R],
        worker_parallel: ParallelConfig | None = None,
    ):
        _log.debug("persisting to Ray cluster")
        self.model_ref = ray.put(model)
        self.function = func
        self.action = ray_invoke_worker.options(num_cpus=inference_worker_cpus(), lk_process=1)

    def map(self, tasks: Iterable[A]) -> Iterator[R]:
        for args in itertools.batched(tasks, 200):
            ret = self.action.remote(self.function, self.model_ref, args)
            yield from ray.get(ret)

    def shutdown(self):
        del self.model_ref


@ray.remote
def ray_invoke_worker(func: Callable[[M, A], R], model: M, args: list[A]) -> list[R]:
    global _worker_log

    try:
        with Task("cluster worker", subprocess=True) as task:
            result = [func(model, arg) for arg in args]
    finally:
        _worker_log.send_task(task)

    return result
