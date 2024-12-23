Data Splitting
==============

.. py:module:: lenskit.splitting

The :mod:`lenskit.splitting` package implements data splitting support for
evaluation.

Output Types
------------

.. autosummary::
    :toctree:

    ~lenskit.splitting.TTSplit

Temporal Splitting
------------------

.. autosummary::
    :toctree:
    :nosignatures:

    ~lenskit.splitting.split_global_time

User-Based Splitting
--------------------

.. autosummary::
    :toctree:
    :nosignatures:

    ~lenskit.splitting.crossfold_users
    ~lenskit.splitting.sample_users
    ~lenskit.splitting.LastFrac
    ~lenskit.splitting.LastN
    ~lenskit.splitting.SampleFrac
    ~lenskit.splitting.SampleN

Record-Based Splitting
----------------------

.. autosummary::
    :toctree:
    :nosignatures:

    ~lenskit.splitting.crossfold_records
    ~lenskit.splitting.sample_records
