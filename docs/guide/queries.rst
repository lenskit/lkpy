.. _queries:

Queries and Operations
======================

.. py:currentmodule:: lenskit.data

To request recommendations from a LensKit pipeline, you supply the pipeline with
a *query* (and, optionally, a list of candidate items).  In the simplest case,
for traditional personalized recommendations, the query just consists of a user
ID.

Queries are represented by the :class:`.RecQuery` class, and can have any or all
of the following components:

- A user ID (:attr:`RecQuery.user_id`)
- A user's historical items (:attr:`RecQuery.user_items`)

.. admonition:: Future Work
    :class: note

    Queries will also be the basis for eventually supporting session-based
    recommendation, etc.

Creating a Query
~~~~~~~~~~~~~~~~

You can create a query in a couple of ways — you can pass named arguments to the
:class:`RecQuery` constructor, or you can call :meth:`RecQuery.create` to
“upgrade” a piece of data into a query using the following rules:

* A query is returned as-is.
* An identifier is used as the user identifier.
* An :class:`ItemList` is used as a list of the user's historical items.

These data types are encompased by the :class:`QueryInput` type.

You can also pass a raw user identifier or item list to the pipeline, as the key
recommendation operations and most components accept a :class:`QueryInput` and
pass it to :meth:`~RecQuery.create` to upgrade to a query.

Invoking Recommenders
~~~~~~~~~~~~~~~~~~~~~

.. py:currentmodule:: lenskit.operations

LensKit provides three *operation* functions to ease calling the recommender for
common operations, like top-*N* recommendation and rating prediction:

- :func:`.recommend`
- :func:`.score`
- :func:`.predict`

These functions take a pipeline and a query input and return the results:

.. code:: python

    rec_list = recommend(pipe, user_id, n=20)

Processing Queries
~~~~~~~~~~~~~~~~~~

.. py:currentmodule:: lenskit.data

When writing a component that uses a query but works on user ID and/or user
history arguments, it should have a ``query`` parameter of type ``QueryInput``,
and pass it to :meth:`RecQuery.create` to obtain a query.  You can declare the
query to be of type ``RecQuery`` so long as you provide query objects to the
operation functions, or configure a pipeline to upgrade the query before it gets
to your component.

When using a query in a component, we recommend using the user history if it is
available, instead of relying on user IDs.  This makes components more flexible
for other sources of user history data.

Expanding Queries in Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :ref:`standard pipelines <standard-pipelines>` include a
:class:`~lenskit.basic.UserTrainingHistoryLookup` that resolves a query and, if
it does not have user history data, looks up the user's historical clicks from
the training data.  Therefore, if your components will be used in the standard
pipeline (or another pipeline configured with this component), you can take a
:class:`RecQuery` as input, and expect it to have a user history if the user is
known in the training data.
