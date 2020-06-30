"""
Code to import commonly-used RecSys data sets into LensKit-compatible data frames.
"""

import os.path
from pathlib import Path
import logging

import pandas as pd
import numpy as np

_log = logging.getLogger(__name__)

__doctest_skip__ = []
if not os.path.exists('data/ml-100k'):
    __doctest_skip__.append('ML100K.*')
if not os.path.exists('data/ml-20m'):
    __doctest_skip__.append('MovieLens.tag_genome')
if not os.path.exists('data/ml-1m.*'):
    __doctest_skip__.append('ML1M.*')
if not os.path.exists('data/ml-10M100K'):
    __doctest_skip__.append('ML10M.*')
    __doctest_skip__.append('MLM.*')


def cached(prop):
    cache = '_cached_' + prop.__name__

    def getter(self):
        val = getattr(self, cache, None)
        if val is None:
            val = prop(self)
            setattr(self, cache, val)
        return val

    getter.__doc__ = prop.__doc__

    return property(getter)


class MovieLens:
    """
    Code for reading current MovieLens data sets, including ML-20M, ML-Latest, and
    ML-Latest-Small.

    Parameters:
        path(str or pathlib.Path): Path to the directory containing the data set.
    """

    def __init__(self, path='data/ml-20m'):
        self.path = Path(path)

    @cached
    def ratings(self):
        """
        The rating table.

        >>> mlsmall = MovieLens('data/ml-latest-small')
        >>> mlsmall.ratings
                user  item  rating   timestamp
        0          1    31     2.5  1260759144
        1          1  1029     3.0  1260759179
        2          1  1061     3.0  1260759182
        3          1  1129     2.0  1260759185
        4          1  1172     4.0  1260759205
        ...
        [100004 rows x 4 columns]
        """

        fn = self.path / 'ratings.csv'
        ratings = pd.read_csv(fn, dtype={
            'movieId': np.int32,
            'userId': np.int32,
            'rating': np.float64,
            'timestamp': np.int32
        })
        ratings.rename(columns={'userId': 'user', 'movieId': 'item'}, inplace=True)
        _log.debug('loaded %s, takes %d bytes', fn, ratings.memory_usage().sum())
        return ratings

    @cached
    def movies(self):
        """
        The movie table, with titles and genres.  It is indexed by movie ID.

        >>> mlsmall = MovieLens('data/ml-latest-small')
        >>> mlsmall.movies
                                                            title                                           genres
        item
        1                                        Toy Story (1995)      Adventure|Animation|Children|Comedy|Fantasy
        2                                          Jumanji (1995)                       Adventure|Children|Fantasy
        3                                 Grumpier Old Men (1995)                                   Comedy|Romance
        4                                Waiting to Exhale (1995)                             Comedy|Drama|Romance
        5                      Father of the Bride Part II (1995)                                           Comedy
        ...
        [9125 rows x 2 columns]
        """

        fn = self.path / 'movies.csv'
        movies = pd.read_csv(fn, dtype={
            'movieId': np.int32,
            'title': np.object,
            'genres': np.object
        })
        movies.rename(columns={'movieId': 'item'}, inplace=True)
        movies.set_index('item', inplace=True)
        _log.debug('loaded %s, takes %d bytes', fn, movies.memory_usage().sum())
        return movies

    @cached
    def links(self):
        """
        The movie link table, connecting movie IDs to external identifiers.  It is indexed
        by movie ID.

        >>> mlsmall = MovieLens('data/ml-latest-small')
        >>> mlsmall.links
                 imdbId  tmdbId
        item
        1        114709     862
        2        113497    8844
        3        113228   15602
        4        114885   31357
        5        113041   11862
        ...
        [9125 rows x 2 columns]
        """

        fn = self.path / 'links.csv'
        links = pd.read_csv(fn, dtype={
            'movieId': np.int32,
            'imdbId': np.int64,
            'tmdbId': pd.Int64Dtype()
        })
        links.rename(columns={'movieId': 'item'}, inplace=True)
        links.set_index('item', inplace=True)
        _log.debug('loaded %s, takes %d bytes', fn, links.memory_usage().sum())
        return links

    @cached
    def tags(self):
        """
        The tag application table, recording user-supplied tags for movies.


        >>> mlsmall = MovieLens('data/ml-latest-small')
        >>> mlsmall.tags
              user  ...   timestamp
        0       15  ...  1138537770
        1       15  ...  1193435061
        2       15  ...  1170560997
        3       15  ...  1170626366
        4       15  ...  1141391765
        ...
        [1296 rows x 4 columns]
        """

        fn = self.path / 'tags.csv'
        tags = pd.read_csv(fn, dtype={
            'movieId': np.int32,
            'userId': np.int32,
            'tag': np.object,
            'timestamp': np.int32
        })
        tags.rename(columns={'userId': 'user', 'movieId': 'item'}, inplace=True)
        _log.debug('loaded %s, takes %d bytes', fn, tags.memory_usage().sum())
        return tags

    @cached
    def tag_genome(self):
        """
        The tag genome table, recording inferred item-tag relevance scores.  This gets returned
        as a wide Pandas data frame, with rows indexed by item ID.

        >>> ml20m = MovieLens('data/ml-20m')
        >>> ml20m.tag_genome
        tag         007  007 (series)  18th century  ...     wwii   zombie  zombies
        item                                         ...
        1       0.02500       0.02500       0.05775  ...  0.03625  0.07775  0.02300
        2       0.03975       0.04375       0.03775  ...  0.01475  0.09025  0.01875
        3       0.04350       0.05475       0.02800  ...  0.01950  0.09700  0.01850
        4       0.03725       0.03950       0.03675  ...  0.01525  0.06450  0.01300
        5       0.04200       0.05275       0.05925  ...  0.01675  0.10750  0.01825
        ...
        [10381 rows x 1128 columns]
        """

        fn = self.path / 'genome-scores.csv'
        tags = pd.read_csv(self.path / 'genome-tags.csv')
        tags = tags.set_index('tagId')
        tags = tags['tag'].astype('category')
        genome = pd.read_csv(fn, dtype={
            'movieId': np.int32,
            'tagId': np.int32,
            'relevance': np.float64
        })
        genome.rename(columns={'userId': 'user', 'movieId': 'item'}, inplace=True)
        genome = genome.join(tags, on='tagId')
        genome = genome.pivot(index='item', columns='tag', values='relevance')
        _log.debug('loaded %s, takes %d bytes', fn, genome.memory_usage().sum())
        return genome


class ML100K:
    """
    The MovieLens 100K data set.  This older data set is in a different format from
    the more current data sets loaded by :class:`MovieLens`.
    """

    def __init__(self, path='data/ml-100k'):
        self.path = Path(path)

    @property
    def available(self):
        "Query whether the data set exists."
        return (self.path / 'u.data').exists()

    @cached
    def ratings(self):
        """
        Return the rating data (from ``u.data``).

        >>> ml = ML100K()
        >>> ml.ratings
               user  item  rating  timestamp
        0       196   242     3.0  881250949
        1       186   302     3.0  891717742
        2        22   377     1.0  878887116
        3       244    51     2.0  880606923
        4       166   346     1.0  886397596
        ...
        [100000 rows x 4 columns]
        """
        fn = self.path / 'u.data'
        ratings = pd.read_csv(fn, sep='\t', header=None,
                              names=['user', 'item', 'rating', 'timestamp'],
                              dtype={'user': np.int32, 'item': np.int32,
                                     'rating': np.float32, 'timestamp': np.int32})
        _log.debug('loaded %s', fn)
        return ratings

    @cached
    def users(self):
        """
        Return the user data (from ``u.user``).

        >>> ml = ML100K()
        >>> ml.users
              age gender     occupation     zip
        user
        1      24      M     technician   85711
        2      53      F          other   94043
        3      23      M         writer   32067
        4      24      M     technician   43537
        5      33      F          other   15213
        ...
        [943 rows x 4 columns]
        """
        fn = self.path / 'u.user'
        users = pd.read_csv(fn, sep='|', header=None,
                            names=['user', 'age', 'gender', 'occupation', 'zip'],
                            dtype={'user': np.int32, 'age': np.int8,
                                   'occupation': 'category'})
        _log.debug('loaded %s', fn)
        return users.set_index('user')

    @cached
    def movies(self):
        """
        Return the user data (from ``u.user``).

        >>> ml = ML100K()
        >>> ml.movies
                                                  title      release  ...  War Western
        item                                                          ...
        1                              Toy Story (1995)  01-Jan-1995  ...    0       0
        2                              GoldenEye (1995)  01-Jan-1995  ...    0       0
        3                             Four Rooms (1995)  01-Jan-1995  ...    0       0
        4                             Get Shorty (1995)  01-Jan-1995  ...    0       0
        5                                Copycat (1995)  01-Jan-1995  ...    0       0
        ...
        [1682 rows x 23 columns]
        """
        fn = self.path / 'u.item'
        genres = [
            'unknown', 'Action', 'Adventure', 'Animation',
            "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western'
        ]
        items = pd.read_csv(fn, sep='|', header=None, encoding='latin1',
                            names=['item', 'title', 'release', 'vidrelease', 'imdb'] + genres)
        _log.debug('loaded %s', fn)
        return items.set_index('item')


class MLM:
    """
    Base classes for ML1M and ML10M.
    """

    def __init__(self, path):
        self.path = Path(path)

    @cached
    def ratings(self):
        """
        Return the rating data (from ``ratings.dat``).

        >>> ml = ML10M()
        >>> ml.ratings
                   user  item  rating  timestamp
        0             1   122     5.0  838985046
        1             1   185     5.0  838983525
        2             1   231     5.0  838983392
        3             1   292     5.0  838983421
        4             1   316     5.0  838983392
        ...
        [10000054 rows x 4 columns]
        """
        fn = self.path / 'ratings.dat'
        ratings = pd.read_csv(fn, sep=':', header=None,
                              names=['user', '_ui', 'item', '_ir', 'rating', '_rt', 'timestamp'],
                              usecols=[0, 2, 4, 6],
                              dtype={'user': np.int32, 'item': np.int32,
                                     'rating': np.float32, 'timestamp': np.int32})
        _log.debug('loaded %s', fn)
        return ratings

    @cached
    def movies(self):
        """
        Return the movie data (from ``movies.dat``).  Indexed by movie ID.

        >>> ml = ML10M()
        >>> ml.movies
                                                            title                                           genres
        item
        1                                        Toy Story (1995)      Adventure|Animation|Children|Comedy|Fantasy
        2                                          Jumanji (1995)                       Adventure|Children|Fantasy
        3                                 Grumpier Old Men (1995)                                   Comedy|Romance
        4                                Waiting to Exhale (1995)                             Comedy|Drama|Romance
        5                      Father of the Bride Part II (1995)                                           Comedy
        ...
        [10681 rows x 2 columns]
        """
        fn = self.path / 'movies.dat'
        movies = pd.read_csv(fn, sep=':', header=None,
                             names=['item', '_ir', 'title', '_tg', 'genres'],
                             usecols=[0, 2, 4],
                             dtype={'item': np.int32})
        movies.set_index('item', inplace=True)
        _log.debug('loaded %s', fn)
        return movies


class ML10M(MLM):
    """
    MovieLens 10M100K data set.
    """
    def __init__(self, path='data/ml-10M100K'):
        super().__init__(path)


class ML1M(MLM):
    """
    MovieLens 1M data set.

    .. note::
       Some documentation examples use ML-10M100K; that is because this class shares implementation
       with the 10M data set.
    """

    def __init__(self, path='data/ml-1m'):
        super().__init__(path)

    @cached
    def users(self):
        """
        Return the movie data (from ``users.dat``).  Indexed by user ID.

        >>> ml = ML1M()
        >>> ml.users
             gender  age    zip
        user
        1         F    1  48067
        2         M   56  70072
        3         M   25  55117
        4         M   45  02460
        5         M   25  55455
        ...
        [6040 rows x 3 columns]
        """
        fn = self.path / 'users.dat'
        users = pd.read_csv(fn, sep=':', header=None,
                            names=['user', '_ug', 'gender', '_ga', 'age',
                                   '_ao', 'occupation', '_oz', 'zip'],
                            usecols=[0, 2, 4, 8],
                            dtype={'user': np.int32, 'gender': 'category', 'age': np.int8,
                                   'timestamp': np.int32})
        users.set_index('user', inplace=True)
        _log.debug('loaded %s', fn)
        return users
