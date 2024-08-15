Data Abstractions
=================

.. py:module:: lenskit.data

The :mod:`lenskit.data` module provides the core data abstractions LensKit uses
to represent recommender system inputs and outputs.

Data Sets
---------

.. autosummary::
    :toctree: .
    :nosignatures:
    :caption: Data Sets

    ~lenskit.data.Dataset
    ~lenskit.data.from_interactions_df
    ~lenskit.data.load_movielens
    ~lenskit.data.load_movielens_df


Item Data
---------

.. autosummary::
    :toctree: .
    :nosignatures:
    :caption: Item Data

    ~lenskit.data.ItemList

User Data
---------

.. autosummary::
    :toctree: .
    :nosignatures:
    :caption: User Data

    ~lenskit.data.UserProfile

Identifiers
-----------

.. autosummary::
    :toctree: .
    :nosignatures:
    :caption: Identifiers

    ~lenskit.data.Vocabulary
