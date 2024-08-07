Data Management
===============

.. module:: lenskit.data

LensKit provides a unified data model for recommender systems data along with
classes and utility functions for working with it, described in this section of
the manual.


.. versionchanged:: 2024.1
    The new :class:`Dataset` class replaces the Pandas data frames
    that were passed to algorithms in the past.  It also subsumes
    the old support for producing sparse matrices from rating frames.

.. _data-model:

Data Model and Key Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LensKit data model consists of **users**, **items**, and **interactions**,
with fields providing additional (optional) data about each of these entities.
The simplest valid LensKit data set is simply a list of user and item
identifiers indicating which items each user has interacted with.  These may be
augumented with ratings, timestamps, or any other attributes.

Data can be read from a range of sources, but ultimately resolves to a
collection of tables (e.g. Pandas :class:`~pandas.DataFrame`) that record user,
item, and interaction data.

.. _data-identifiers:

Identifiers
-----------

Users and items have two identifiers:

* The *identifier* as presented in the original source table(s).  It appears in
  LensKit data frames as ``user_id`` and ``item_id`` columns.  Identifiers can
  be integers, strings, or byte arrays.
* The *number* assigned by the dataset handling code.  This is a 0-based
  contiguous user or item number that is suitable for indexing into arrays or
  matrices, a common operation in recommendation models.  In data frames, this
  appears as a ``user_num`` or ``item_num`` column.  It is the only
  representation supported by NumPy and PyTorch array formats.

  User and item numbers are assigned based on sorted identifiers in the initial
  data source, so reloading the same data set will yield the same numbers.
  Loading a subset, however, is not guaranteed to result in the same numbers, as
  the subset may be missing some users or items.

  Methods that add additional users or items will assign numbers based on the
  sorted identifiers that do not yet have numbers.

Identifiers and numbers can be mapped to each other with the user and item
*vocabularies* (:attr:`~Dataset.users` and :attr:`~Dataset.items`, see the
:class:`~lenskit.data.vocab.Vocabulary` class).

.. autodata:: EntityId

.. _dataset:

Dataset Abstraction
~~~~~~~~~~~~~~~~~~~

The LensKit :class:`Dataset` class is the standard LensKit interface to datasets
for training, evaluation, etc. Trainable models and components expect a dataset
instance to be passed to :meth:`~lenskit.algorithms.Recommender.fit`.  It is an
abstract class with implementations covering various scenarios.

.. autoclass:: Dataset

Creating Datasets
~~~~~~~~~~~~~~~~~

Several functions can create a :class:`Dataset` from different input data sources.

.. autofunction:: from_interactions_df

Loading Common Datasets
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_movielens

Vocabularies
~~~~~~~~~~~~

LensKit uses *vocabularies* to record user/item IDs, tags, terms, etc. in a way
that facilitates easy mapping to 0-based contiguous indexes for use in matrix
and tensor data structures.

.. autoclass:: Vocabulary

User and Item Data
~~~~~~~~~~~~~~~~~~

The :mod:`lenskit.data` package also provides various classes for representing
user and item data.

User Profiles
-------------

.. autoclass:: UserProfile

Item Lists
----------

LensKit uses *item lists* to represent collections of items that may be scored,
ranked, etc.

.. autoclass:: ItemList

.. autoclass:: HasItemList

User-Item Data Tables
~~~~~~~~~~~~~~~~~~~~~

.. module:: lenskit.data.tables

.. autoclass:: NumpyUserItemTable
.. autoclass:: TorchUserItemTable

Dataset Implementations
~~~~~~~~~~~~~~~~~~~~~~~

.. module:: lenskit.data.dataset

Matrix Dataset
--------------

The :class:`MatrixDataset` provides an in-memory dataset implementation backed
by a ratings matrix or implicit-feedback matrix.

.. autoclass:: MatrixDataset
    :no-members:

Lazy Dataset
------------

The lazy data set takes a function that loads a data set (of any type), and
lazily uses that function to load an underlying data set when needed.

.. autoclass:: LazyDataset
    :no-members:
    :members: delegate
