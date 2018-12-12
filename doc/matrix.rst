Matrix Utilities
----------------

.. module:: lenskit.matrix

We have some matrix-related utilities, since matrices are used so heavily in recommendation
algorithms.

Building Ratings Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sparse_ratings
.. autoclass:: RatingMatrix

Compressed Sparse Row Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use CSR-format sparse matrices in quite a few places. Since SciPy's sparse matrices are not
directly usable from Numba, we have implemented a Numba-compiled CSR representation that can
be used from accelerated algorithm implementations.

.. autofunction:: csr_from_coo
.. autofunction:: csr_from_scipy
.. autofunction:: csr_to_scipy
.. autofunction:: csr_rowinds
.. autofunction:: csr_save
.. autofunction:: csr_load

.. autoclass:: CSR
    :members:
