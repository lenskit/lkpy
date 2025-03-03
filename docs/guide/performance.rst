Performance Tips
================

LensKit strives to provide pretty good performance (in terms of computation speed), but
sometimes it needs a little nudging.

.. todo::

    Update for 2025.1.

.. note::

    If you are implementing an algorithm, see the `implementation tips`_ for information
    on good performance.

.. _implementation tips: impl-tips.html

Quick Tips
----------

* Use Conda-based Python, with ``tbb`` installed.
* When using MKL, set the ``MKL_THREADING_LAYER`` environment variable to ``tbb``, so both
  MKL and LensKit will use TBB and can coordinate their thread pools.
* Use ``LK_NUM_PROCS`` if you want to control LensKit's batch prediction and recommendation
  parallelism, and ``NUMBA_NUM_THREADS`` to control its model training parallelism.

We generally find the best performance using MKL with TBB throughout the stack on Intel
processors.  If both LensKit's Numba-accelerated code and MKL are using TBB, they will
coordinate their thread pools to coordinate threading levels.

If you are **not** using MKL (Apple Silicon, maybe also AMD processors), we recommend
controlling your BLAS parallelism.  For OpenBLAS, how you control this depends on how
OpenBLAS was built, whether Numba is using OpenMP or TBB, and whether you are training
or evaluating the model.

When LensKit starts (usually at model training time), it will check your runtime environment
and log warning messages if it detects problems.  During evaluation, it also makes a
best-effort attempt, through `threadpoolctl`_, to disable nested parallelism when running
a parallel evaluation.

.. _threadpoolctl: https://github.com/joblib/threadpoolctl

Controlling Parallelism
-----------------------

LensKit has two forms of parallelism.  Algorithm training processes can be parallelized
through a number of mechanisms:

* Our own parallel code uses Numba, which in turn uses TBB (preferred) or OpenMP.  The
  thread count is controlled by ``NUMBA_NUM_THREADS``.
* The BLAS library may parallelize underlying operations using its threading library.
  This is usually OpenMP; MKL also supports TBB, but unlike Numba, it defaults to
  OpenMP even if TBB is available.
* Underlying libraries such as TensorFlow and scikit-learn may provide their
  own parallelism.

The LensKit `batch functions`_ use Python ``multiprocessing``, and their concurrency
level is controlled by the ``LK_NUM_PROCS`` environment variable.  The default number
of processes is one-half the number of cores as reported by :py:func:`multiprocessing.cpu_count`.
The batch functions also set the thread count for some libraries within the worker
procesess, to prevent over-subscribing the CPU.  Right now, the worker will configure
Numba and MKL.  In the rest of this section, this will be referred to as the ‘inner
thread count’.

The thread count logic is controlled by :py:func:`lenskit.parallel.initialize`,
and works as follows:

* If ``LK_NUM_PROCS`` is an integer, the batch functions will use the specified number
  of processes, and with 1 inner thread.
* If ``LK_NUM_PROCS`` is a comma-separated pair of integers (e.g. ``8,4``), the batch
  functions will use the first number for the process count and the second number as
  the inner thread count.  This **overrides** ``NUMBA_NUM_THREADS``, unless it is larger
  than ``NUMBA_NUM_THREADS``.
* If ``LK_NUM_PROCS`` is not set, the batch functions use half the number of cores as
  the process count and 2 as the inner thread count (unless ``NUMBA_NUM_THREADS`` is
  set to 1 in the environment).

.. _batch functions: batch.html

Other Notes
-----------

* Batch parallelism **disables** TensorFlow GPUs in the worker threads.  This is fine,
  because GPUs are most useful for model training; multiple worker processes competing
  for the GPU causes problems.
