Data Abstractions
=================

.. py:module:: lenskit.data

The :mod:`lenskit.data` module provides the core data abstractions LensKit uses
to represent recommender system inputs and outputs.

.. toctree::
    :maxdepth: 1

    data-types

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


Terms and Identifiers
---------------------

.. autosummary::
    :toctree: .
    :nosignatures:
    :caption: Terms and Identifiers

    ~lenskit.data.Vocabulary

See also:

* :py:class:`lenskit.data.types.EntityId`
