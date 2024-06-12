Parallel Execution
------------------

.. py:module:: lenskit.parallel

LensKit supports various forms of parallel execution, each with an environment
variable controlling its :

- :doc:`Batch operations <batch>` using :ref:`multi-process execution <parallel-model-ops>`.
- Parallel model training.  For most models provided by LensKit, this is usually
  implemented using PyTorch JIT parallelism (:func:`torch.jit.fork`).
- Parallel computation in the various backends (BLAS, MKL, Torch, etc.).

Other models compatible with LensKit may use their own parallel processing logic.

Configuring Parallelism
~~~~~~~~~~~~~~~~~~~~~~~

LensKit provides 4 knobs for configuring parallelism, each of which has a
corresponding environment variable and parameter to :py:func:`initialize`.  The
environment variables are:

.. envvar:: LK_NUM_PROCS

    The number of processes to use for batch operations.  Defaults to the number
    of CPUs or 4, whichever is lower.

.. envvar:: LK_NUM_THREADS

    The number of threads to use for parallel model building.  Defaults to the
    number of CPUs or 8, whichever is smaller.

    This number is passed to :func:`torch.set_num_interop_threads` to set up the
    Torch JIT thread count.

.. envvar:: LK_NUM_BACKEND_THREADS

    The number of threads to be used by backend compute engines.  Defaults to up
    to 4 backend threads per training thread, depending on the capacity of the
    machine::

        max(min(NCPUS // LK_NUM_THREADS, 4), 1)

    This is passed to :func:`torch.set_num_threads` (to control PyTorch internal
    parallelism), and to the underlying BLAS layer (via `threadpoolctl`_).

.. envvar:: LK_NUM_CHILD_THREADS

    The number of backend threads to be used in worker processes spawned by
    batch evaluation.  Defaults to 4 per process, capped by the number of CPUs
    available::

        max(min(NCPUS // LK_NUM_PROCS, 4), 1)

    Workers have both the process and thread counts set to 1.

.. _threadpoolctl: https://github.com/joblib/threadpoolctl

.. autofunction:: initialize

.. autofunction:: ensure_parallel_init

.. _parallel-model-ops:

Parallel Model Ops
~~~~~~~~~~~~~~~~~~

LensKit uses a custom API wrapping  :py:class:`multiprocessing.pool.Pool` to
parallelize batch operations (see :py:mod:`lenskit.batch`).

The basic idea of this API is to create an *invoker* that has a model and a function,
and then passing lists of argument sets to the function::

    with invoker(model, func):
        results = list(func.map(args))

The model is persisted into shared memory to be used by the worker processes.
PyTorch tensors, including those on CUDA devices, are shared.

LensKit users will generally not need to directly use parallel op invokers, but
if you are implementing new batch operations with parallelism they are useful.
They may also be useful for other kinds of analysis.

.. autofunction:: invoker

.. autoclass:: ModelOpInvoker
    :members:

Logging and Progress
~~~~~~~~~~~~~~~~~~~~

Multi-process op invokers automatically set up logging and progress reporting to
work across processes using the :py:mod:`manylog` package.  Op invokers can also
report the progress of queued jobs to a :py:class:`progress_api.Progress`.

.. autofunction:: invoke_progress

Computing Work Chunks
~~~~~~~~~~~~~~~~~~~~~

.. py:module:: lenskit.parallel.chunking

The :py:class:`WorkChunks` class provides support for dividing work into chunks for
parallel processing, particularly for model training.

.. autoclass:: WorkChunks
