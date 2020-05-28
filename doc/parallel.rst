Parallel Execution
------------------

.. py:module:: lenskit.util.parallel

LensKit uses :py:class:`concurrent.futures.ProcessPoolExecutor` to paralellize batch
operations (see :py:mod:`lenskit.batch`).

The basic idea of this API is to create an *invoker* that has a model and a function,
and then passing lists of argument sets to the function::

    with invoker(model, func):
        results = list(func.map(args))

The model is persisted into shared memory to be used by the worker processes.


.. autofunction:: invoker

.. autofunction:: proc_count

.. autoclass:: ModelOpInvoker
    :members:
