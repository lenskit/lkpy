.. _std-amazon:

Amazon Data
~~~~~~~~~~~

Julian McAuley's group at UCSD has collected several datasets of Amazon ratings
and reviews, which LensKit supports.

These sets include:

- `2023 Ratings and Reviews <https://amazon-reviews-2023.github.io/>` (recommended for new projects)
- `2018 Ratings and Reviews <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/>` (V2, from 2018)
- `2014 Ratings and Reviews <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html>`

Currently, LensKit supports reading the rating CSV files with
:func:`lenskit.data.load_amazon_ratings`.  The command line tools also support Amazon data.

.. note::

    Several data files from 2014 and 2018 contain duplicates, which currently are not supported.

Data Model
----------

The ratings are in a standard `rating` interaction on the dataset with a `timestamp` attribute.
