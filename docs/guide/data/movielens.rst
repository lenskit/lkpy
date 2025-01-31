.. _std-movielens:

MovieLens Data
~~~~~~~~~~~~~~

The MovieLens_ data sets are a widely-used set of movie rating datasets,
available from the `GroupLens dataset collection`_.  The core of these data sets
is matrix of user-provided 5-star ratings of movies, along with movie metadata
such as titles and IMDB links.  Some sets include user demographics as well, and
others include various forms of tag data.

.. _MovieLens: https://movielens.org
.. _GroupLens dataset collection: https://grouplens.org/datasets/movielens

Loading MovieLens Data
======================

The :func:`~lenskit.data.load_movielens` function will load any published
MovieLens dataset, constructing a LensKit :class:`~lenskit.data.Dataset` with
its contents. This dataset can then be split, saved in LensKit native format,
used to train models and pipelines, etc.

This function automatically detects which MovieLens dataset is being loaded,
and can load them from either the Zip archives published by GroupLens or from
a directory where the archive has been unpacked.

MovieLens Data Model
~~~~~~~~~~~~~~~~~~~~

The MovieLens loader loads the data into the standard ``user`` and ``item``
entities, with a ``rating`` interaction class storing the user-provided ratings.
The items have the following attributes:

``title``
    The movie title.
``genres``
    A list of genres for this movie.
``tag_counts``
    A sparse vector attribute storing the number of times each tag has been
    applied to this movie.  It is a summary of the ``tags`` data provided by
    MovieLens.  The tag names themselves are on the attribute's
    :attr:`~lenskit.data.AttributeSet.names`.
``tag_genome``
    A vector attribute storing the relevance values from the *tag genome*
    :cite:p:`vigTagGenomeEncoding2012`, when it is available (ML20M and 25M).

For most data sets, there are no user attributes; ML100K and ML1M have
``gender``, ``age``, and ``zip_code`` attributes.  See the MovieLens data
documentation for details on these.

Ratings have two attributes: ``rating`` and ``timestamp``.  The timestamps
are parsed into Arrow/NumPy/Pandas timestamps.
