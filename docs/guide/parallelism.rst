.. _parallelism:

Parallel Processing
===================

.. py:currentmodule:: lenskit.parallel

LensKit supports various forms of parallel execution, each with an environment
variable controlling its:

- :doc:`Batch operations <batch>` using either a thread pool (on free-threaded
  builds of Python 3.14 and later) or Ray (with the ``lenskit[ray]`` extra
  installed).
- Parallel model training.  For most models provided by LensKit, such
  parallelism uses Rayon in the Rust extension module.
- Parallel inference, using Rayon in the Rust extension module.
- Parallel computation in the various backends (BLAS, MKL, Torch, etc.), for
  both training and inference.

.. versionchanged:: 2026.1

    LensKit no longer uses multiprocessing or PyTorch JIT parallelism, and the
    configuration options in the next section have been revised.

.. tip::

    LensKit supports multithreaded operation with free-threaded Python 3.14 or
    newer.  You can install such a Python with ``uv``:

    .. code:: console

        uv venv -p 3.14t

    If your Python is not free-threaded and you want to parallelize batch
    inference, you will need to also install Ray, and performance will likely
    not be quite as good.

.. _parallel-config:

Configuring Parallelism
~~~~~~~~~~~~~~~~~~~~~~~

LensKit provides several controls to configure its use of parallelism, each of
which has a corresponding environment variable and parameter in the
``[parallel]`` section of :file:`lenskit.toml`, defined by
:class:`~lenskit.config.ParallelSettings`.

The default settings are based on the number of CPUs available on your system,
capped to reduce the risk of runaway processes hogging an entire machine.  Due
to multiple multithreading mechanisms, a precise limit is difficult to specify,
but with default settings LensKit's usage will stay under about 32 CPU cores,
and will rarely use that many (depending on the components used).  LensKit also
tries to account for operating system CPU limits when detecting the number of
available cores.

Component implementations and support code that wants to make sure that
parallelism is properly configured should call :py:func:`ensure_parallel_init`
prior to performing any possibly-parallel computation.  This function
resolves the parallelism configuration (including any default values)
and sets appropriate thread pool limits on compute backends.

The configuration settings and their default values are:

.. currentmodule:: lenskit.config
.. _ray: https://docs.ray.io

.. envvar:: LK_NUM_THREADS

    The number of threads LensKit will use in its own multithreaded code.  This
    primarily controls the size of the Rayon thread pool used by the Rust
    accelerator, but will also be respected by future training code built on
    free-threading or other thread pool mechanisms.

    The default is 8 or the number of available CPU cores, whichever is lower.

    Configuration option: :attr:`~ParallelSettings.num_threads`.

    .. note::

        Only the k-NN models currently use this at inference time.  Several
        models use it at training time.

.. envvar:: LK_NUM_BATCH_JOBS

    The number of concurrent batch inference tasks.  This used by the pipeline
    batch-run code (see :ref:`batch`) to size the inference thread pool or to
    limit the number of concurrently-executing Ray tasks.

    The default is :attr:`~ParallelSettings.num_threads` if Python is
    free-threaded or Ray is enabled (see :envvar:`LK_USE_RAY`), or 1 on
    non-free-threaded Python without Ray.

    Configuration option: :attr:`~ParallelSettings.num_batch_jobs`.

.. envvar:: LK_USE_RAY

    If set to 1, use Ray_ instead of a thread pool for batch inference.  Requires Ray to be
    installed (use the Ray extra, ``lenskit[ray]``, to depend on a compatible
    version of ray).

    Configuration option: :attr:`~ParallelSettings.use_ray`.

.. envvar:: LK_NUM_BACKEND_THREADS

    The number of threads to be used by backend compute engines such as BLAS.
    The configured or derived value is passed to :func:`torch.set_num_threads`
    (to control PyTorch internal parallelism), and to the underlying BLAS layer
    (via `threadpoolctl`_).

    The default is to use up to 4 backend threads per training thread, depending
    on the capacity of the machine::

        max(min(NCPUS // LK_NUM_THREADS, 4), 1)

    Explicitly set this variable to -1 to keep LensKit from trying to configure
    compute backend thread pools and leave them with their default settings.

    Configuration option: :attr:`~ParallelSettings.num_backend_threads`.

    .. note::

        If the ``OPENBLAS_NUM_THREADS`` or ``MKL_NUM_THREADS`` environment
        variables are set, LensKit does not configure the BLAS thread pool.  It
        still tries to configure the PyTorch intra-op thread count.

.. _threadpoolctl: https://github.com/joblib/threadpoolctl

Debugging Parallelism and Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``lenskit doctor`` CLI command inspects the configured environment,
including parallelism configuration.
