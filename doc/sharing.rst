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

Persistence API
---------------

These functions are used for internal LensKit infrastructure code to persist models into
shared memory for parallel processing.

.. autofunction:: persist

.. autoclass:: PersistedModel
    :members:
