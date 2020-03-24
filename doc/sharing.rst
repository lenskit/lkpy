Model Sharing
=============

.. py:module:: lenskit.sharing

The :py:mod:`lenskit.sharing` module provides utilities for managing models and sharing them
between processes, particularly for the  multiprocessing in :py:mod:`lenskit.batch`.


Sharing Mode
-------------

The only piece **algorithm developers** usually need to directly handle is the concept of
'sharing mode' when implementing custom pickling logic.  To save space, it is reasonable to
exclude intermediate data structures, such as caches or inverse indexes, from the pickled
representation of an algorithm, and reconstruct them when the model is loaded.

However, LensKit's multi-process sharing *also* uses pickling to capture the object state
while using shared memory for :py:class:`numpy.ndarray` objects.  In these cases, the structures
should be pickled, so they can be shared between model instances.

To support this, we have the concept of *sharing mode*.  Code that excludes objects when
pickling should call :py:func:`in_share_context` to determine if that exclusion should
actually happen.

.. autofunction:: in_share_context

.. autofunction:: sharing_mode


Model Store API
---------------

Model stores handle persisting models into shared memory, cleaning up shared memory, and
making objects available to other classes.

.. autofunction:: get_store

.. autoclass:: BaseModelStore
    :members:
    :show-inheritance:

.. autoclass:: BaseModelClient
    :members:
    :show-inheritance:

Model Store Implementations
---------------------------

We provide several model store implementations.

Joblib-based
~~~~~~~~~~~~

.. py:module:: lenskit.sharing.joblib

The :joblib-based store works on any supported platform and Python version.  It uses
Joblib's memory-mapped Pickle extension to store models on disk and use their storage to back
memory-mapped views of major data structures.

.. autoclass:: JoblibModelStore
    :show-inheritance:
.. autoclass:: JoblibModelClient
    :show-inheritance:

Shared Memory
~~~~~~~~~~~~~

.. py:module:: lenskit.sharing.sharedmem

This store uses Python 3.8's :py:mod:`multiprocessing.shared_memory` module, along with out-of-band
buffer support in Pickle Protcol 5, to pass model data through shared memory.

.. autoclass:: SHMModelStore
    :show-inheritance:
.. autoclass:: SHMClient
    :show-inheritance:
