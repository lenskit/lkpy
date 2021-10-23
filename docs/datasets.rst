Loading Data
============

LensKit can work with any data in a :py:class:`pandas.DataFrame` with the expected
columns.  LensKit algorithms expect a ``ratings`` frame to contain the following
columns (in any order):

* ``user``, containing user identifiers.  No requirements are placed on user IDs —
  if an algorithm requires something specific, such as contiguous 0-based identifiers
  for indexing into an array — it will use a :py:class:`pandas.Index` to map them.
* ``item``, containing item identifiers. The same comments apply as for ``user``.
* ``rating``, containing user ratings (if available).  Implicit-feedback code will
  not require ratings.

‘Rating’ data can contain other columns as well, and is a catch-all for any user-item
interaction data.  Algorithms will document any non-standard columns they can make
use of.

:py:meth:`lenskit.algorithms.Recommender.fit` can also accept additional data objects
as keyword arguments, and algorithms that wrap other algorithms will pass this data
through unchanged.  Algorithms ignore extra data objects they receive.  This allows
you to build algorithms that train on data besides user-item interactions, such as
user metadata or item content.

Data Loaders
------------

.. module:: lenskit.datasets

The :py:mod:`lenskit.datasets` module provides utilities for reading a variety
of commonly-used LensKit data sets.  It does not package or automatically
download them, but loads them from a local directory where you have unpacked
the data set.  Each data set class or function takes a ``path`` parameter
specifying the location of the data set.

The normal mode of operation for these utilities is to provide a class for the
data set; this class then exposes the data set's data as attributes.  These
attributes are cached internally, so e.g. accessing :py:attr:`MovieLens.ratings`
twice will only load the data file once.

These data files have normalized column names to fit with LensKit's general
conventions.  These are the following:

- User ID columns are called ``user``.
- Item ID columns are called ``item``.
- Rating columns are called ``rating``.
- Timestamp columns are called ``timestamp``.

Other column names are unchanged.  Data tables that provide information about
specific things, such as a table of movie titles, are indexed by the relevant
ID (e.g. :py:attr:`MovieLens.ratings` is indexed by ``item``).

Data sets supported:

* :class:`MovieLens`
* :class:`ML100K`
* :class:`ML1M`
* :class:`ML10M`

MovieLens Data Sets
-------------------

The GroupLens research group provides several data sets extracted from the
MovieLens service :cite:p:`Harper2015-cx`.
These can be downloaded from https://grouplens.org/datasets/movielens/.

.. autoclass:: MovieLens
    :members:

.. autoclass:: ML100K
    :members:

.. autoclass:: ML1M
    :inherited-members:
    :members:

.. autoclass:: ML10M
    :inherited-members:
    :members:
