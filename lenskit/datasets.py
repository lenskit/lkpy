"""
Code to import commonly-used RecSys data sets into LensKit-compatible data frames.
"""

from pathlib import Path
import logging

import pandas as pd
import numpy as np

_log = logging.getLogger(__name__)


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

        >>> mlsmall = MovieLens('ml-latest-small')
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

        >>> mlsmall = MovieLens('ml-latest-small')
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

        >>> mlsmall = MovieLens('ml-latest-small')
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


        >>> mlsmall = MovieLens('ml-latest-small')
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

        >>> ml = ML100K('ml-100k')
        >>> ml.ratings              #doctest: +SKIP
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

        >>> ml = ML100K('ml-100k')
        >>> ml.users                #doctest: +SKIP
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
        >>> ml.ratings      #doctest: +SKIP
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
        >>> ml.movies       #doctest: +SKIP
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
        >>> ml.users        #doctest: +SKIP
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


class BookCrossing:
    """
    Class for reading the BookCrossing data set.

    Parameters:
        path(str or pathlib.Path): Path to the directory containing the data set.
    """

    def __init__(self, path='data/bx'):
        self.path = Path(path)

    @cached
    def feedback(self):
        """
        Read the BookCrossing ratings.  This reads all ratings, including implicit
        feedback (rating value of 0).

        >>> bx = BookCrossing()
        >>> bx.feedback        #doctest: +SKIP
                   user         item  rating
        0        276725   034545104X       0
        1        276726   0155061224       5
        2        276727   0446520802       0
        3        276729   052165615X       3
        4        276729   0521795028       6
        ...
        [1149780 rows x 3 columns]
        """

        fn = self.path / 'BX-Book-Ratings.csv'
        ratings = pd.read_csv(fn, sep=';', encoding='latin1')
        ratings.rename(columns={'User-ID': 'user', 'ISBN': 'item', 'Book-Rating': 'rating'},
                       inplace=True)
        # clean up some stray characters in ISBNs
        ratings['item'] = ratings['item'].str.replace(r'[^0-9A-Za-z]', '').str.upper()
        _log.debug('loaded %s', fn)
        return ratings

    @property
    def ratings(self):
        """
        Read the BookCrossing ratings, returning only the explicit-feedback ratings.

        >>> bx = BookCrossing()
        >>> bx.ratings        #doctest: +SKIP
                   user         item  rating
        1        276726   0155061224       5
        3        276729   052165615X       3
        4        276729   0521795028       6
        6        276736   3257224281       8
        7        276737   0600570967       6
        ...
        [433671 rows x 3 columns]
        """
        ratings = self.feedback
        return ratings[ratings.rating > 0]
