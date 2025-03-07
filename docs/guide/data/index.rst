.. _datasets:
.. _data-api:

Datasets
========

.. py:currentmodule:: lenskit.data

LensKit provides a unified data model for recommender systems data along with
classes and utility functions for working with it, described in this section of
the manual.

.. versionchanged:: 2025.1
    The new :class:`~lenskit.data.Dataset` class replaces the Pandas data frames
    that were passed to algorithms in the past.  It also subsumes
    the old support for producing sparse matrices from rating frames.

Getting started with the dataset is fairly straightforward:

    >>> from lenskit.data import load_movielens
    >>> mlds = load_movielens('data/ml-latest-small')
    >>> mlds.item_count
    9125

You can then access the data from the various methods of the :class:`Dataset` class.
For example, if you want to get the ratings as a data frame:

    >>> mlds.interaction_matrix(format='pandas', field='rating')
            user_num  item_num  rating
    0              0        30     2.5
    1              0       833     3.0
    2              0       859     3.0
    3              0       906     2.0
    4              0       931     4.0
    ...
    [100004 rows x 3 columns]

Or obtain item statistics:

    >>> mlds.item_stats()
             record_count  user_count  ...           first_time            last_time
    item_id                           ...
    1                 247         247  ...  1996-03-30 19:00:13  2016-10-06 19:55:11
    2                 107         107  ...  1996-03-30 19:12:30  2016-08-01 17:42:33
    3                  59          59  ...  1996-06-05 06:19:04  2016-08-16 22:07:21
    4                  13          13  ...  1996-06-10 16:45:35  2004-07-27 06:14:12
    5                  56          56  ...  1996-04-14 14:23:59  2016-08-16 22:15:47
    ...
    [9125 rows x 7 columns]

.. _data-model:

Data Model and Key Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LensKit data model, detailed in :ref:`data-model` consists of **entities**
(often *users* and *items*) and **interactions**, with attributes providing
additional (optional) data about each of these entities. The simplest valid
LensKit data set is simply a list of user and item identifiers indicating which
items each user has interacted with.  These may be augmented with ratings,
timestamps, or any other attributes.

Data can be read from a range of sources, but ultimately resolves to a
collection of tables (e.g. Pandas :class:`~pandas.DataFrame`) that record user,
item, and interaction data.

.. _data-identifiers:

Identifiers
-----------

Users and items have two identifiers:

* The *identifier* as presented in the original source table(s).  It appears in
  LensKit data frames as ``user_id`` and ``item_id`` columns.  Identifiers can
  be integers, strings, or byte arrays, and are represented in LensKit by the
  :data:`~lenskit.data.ID` type.

* The *number* assigned by the dataset handling code.  This is a 0-based
  contiguous user or item number that is suitable for indexing into arrays or
  matrices, a common operation in recommendation models.  In data frames, this
  appears as a ``user_num`` or ``item_num`` column.  It is the only
  representation supported by NumPy and PyTorch array formats.

  User and item numbers are assigned based on identifiers in the initial data
  source.  Adding all entities at once, or using one of the standard loaders,
  will sort the identifiers before assigning numbers, so reloading the same data
  set will yield the same numbers. Loading a subset, however, is not guaranteed
  to result in the same numbers, as the subset may be missing some users or
  items.

  Adding additional users or items to a data set builder will assign numbers
  based on the sorted identifiers that do not yet have numbers.

Identifiers and numbers can be mapped to each other with the user and item
*vocabularies* (:attr:`~Dataset.users` and :attr:`~Dataset.items`, see the
:class:`Vocabulary` class).

.. _dataset:

Dataset Abstraction
~~~~~~~~~~~~~~~~~~~

The LensKit :class:`Dataset` class is the standard LensKit interface to datasets
for training, evaluation, etc. Trainable models and components expect a dataset
instance to be passed to :meth:`~lenskit.pipeline.Component.train`.

Datasets provide several views of different aspsects of a dataset, documented in
more detail in the :class:`reference documentation <Dataset>`.  These include:

*   Sets of known user and item identifiers, through :class:`Vocabulary` objects
    exposed through the :attr:`Dataset.users` and :attr:`Dataset.items`
    properties.
*   Access to the entities and relationships (including interactions) defined in
    the dataset.

Analyzing Interactions
~~~~~~~~~~~~~~~~~~~~~~

:class:`Dataset` allows client code to obtain *interactions* between entities
(such as users rating items), or other inter-entity relationships, in a variety
of formats (including Pandas data frames and SciPy or PyTorch sparse matrices).
The :class:`RelationshipSet` and :class:`MatrixRelationshipSet` classes provide
the primary interfaces to these capabilities.

.. _interaction-stats:

Interaction Statistics
----------------------

Datasets also provide cached access to various statistics of the entities
involved in an interaction class.  These are currently exposed through
:meth:`MatrixRelationshipSet.row_stats` and
:meth:`~MatrixRelationshipSet.col_stats`; for convenience, the statistics from
the default interaction class are available on :meth:`Dataset.user_stats` and
:meth:`Dataset.item_stats`.

These statistics include:

``count``
    The total number of relationships for the entity.
``record_count``
    The number of relationship or interaction records for the entity.  This is
    equal to ``count``, unless the relationship type has a ``count`` attribute,
    in which case this attribute is the number of records and ``count`` is the
    total number of interactions.
``<other>_count``
    The number of distinct entities of type <other> this entity has interacted
    with.  For example, the user statistics of a normal user-item interaction
    type will have an ``item_count`` column.
``rating_count``
    The number of explicit rating values (only defined if the interaction type
    has a ``rating`` attribute).
``mean_rating``
    The mean rating provided by or for this entity (only defined if the interaction
    type has a ``rating`` attribute).
``first_time``
    The first recorded timestamp for this entity's interactions (only defined if
    the interaction type has a ``timestamp`` attribute).
``last_time``
    The last recorded timestamp for this entity's interactions (only defined if
    the interaction type has a ``timestamp`` attribute).

Creating Datasets
~~~~~~~~~~~~~~~~~

Several functions and classes can create a :class:`Dataset` from different input
data sources.

.. autosummary::
    DatasetBuilder
    from_interactions_df

Loading Common Datasets
~~~~~~~~~~~~~~~~~~~~~~~

LensKit also provides support for loading several common data sets directly from
their source files.

.. autosummary::
    load_movielens

Saving Datasets
~~~~~~~~~~~~~~~

LensKit has a native dataset format to which datasets can be saved and loaded.
This format fully represents the internal data structures.  See the following
methods to use it:

.. autosummary::
    Dataset.save
    Dataset.load
    DatasetBuilder.save

.. admonition:: Compatibility
    :class: note

    The LensKit native dataset format code maintains the following Compatibility
    guarantees:

    -   LensKit can read datasets saved with any earlier minor version in the
        same major-version series (e.g. 2025.2 can read from 2025.1).
    -   LensKit can usually read datasets saved with a later minor version, but
        this is not fully guaranteed.
    -   LensKit will read datasets saved with any prior version on a best-effort
        basis.  We may in the future upgrade this to guarantee full backwards
        compatibility for reading older dataset versions.
