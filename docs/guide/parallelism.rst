.. _parallelism:

Parallel Processing
===================

.. py:currentmodule:: lenskit.parallel

LensKit supports various forms of parallel execution, each with an environment
variable controlling its:

- :doc:`Batch operations <batch>` using :ref:`multi-process execution
  <parallel-model-ops>`.
- Parallel model training.  For most models provided by LensKit, such
  parallelism uses Rayon in the Rust extension module or PyTorch JIT parallelism
  (:func:`torch.jit.fork`) in Python code.
- Parallel inference, using Rayon in the Rust extension module.
- Parallel computation in the various backends (BLAS, MKL, Torch, etc.), for
  both training and inference.

.. _parallel-config:

Configuring Parallelism
~~~~~~~~~~~~~~~~~~~~~~~

LensKit provides 4 knobs for configuring parallelism, each of which has a
corresponding environment variable and parameter to :py:func:`initialize` (the
parameters take precedence over environment variables). Component
implementations and support code that wants to make sure that parallelism is
properly configured should call :py:func:`ensure_parallel_init` prior to
performing any parallelizable computation.

The environment variables and their defaults are:

.. envvar:: LK_NUM_PROCS

    The number of processes to use for batch operations.  Defaults to the number
    of CPUs or 4, whichever is lower.

.. envvar:: LK_NUM_THREADS

    The number of threads to use for LensKit's explicit parallelism (model
    building and inference).  Defaults to the number of CPUs or 8, whichever is
    smaller.

    This number is passed to :func:`torch.set_num_interop_threads` to set up the
    Torch JIT thread count, and is used to configure the Rayon thread pool used
    by the Rust acceleration module.

    .. note::

        Only the k-NN models currently use this at inference time.

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

The number of CPUs (``NCPUS``) is determined by the function
:py:func:`effective_cpu_count`.

.. _threadpoolctl: https://github.com/joblib/threadpoolctl

.. parallel-protecting:

Protecting Scripts for Multiprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any scripts that use LensKit's process-based parallelism support, even
indirectly, must be **import-protected**: that is, the script must not directly
do its work when run, but should define functions and call a ``main`` function
when run as a script, with a block like this at the end of the file::

    def main():
        # do the actual work

    if __name__ == '__main__':
        main()

If you are using the batch functions from a Jupyter notebook, you should be fine
— the Jupyter programs are appropriately protected.

.. _parallel-model-ops:

Parallel Model Ops
~~~~~~~~~~~~~~~~~~

LensKit uses a custom API wrapping :py:class:`multiprocessing.pool.Pool` to
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

.. note::

    Client code generally does not need to directly use this facility.  We are
    also exploring deprecating the internal parallelism support in favor of Ray_.

.. _Ray: https://docs.ray.io

Debugging Parallelism and Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``lenskit doctor`` CLI command inspects the configured environment,
including parallelism configuration.
