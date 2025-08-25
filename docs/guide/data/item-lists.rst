Item Lists and Collections
==========================

.. currentmodule:: lenskit.data

Throughout its data handling, components, and evaluation metrics, LensKit uses
an abstraction of an “item list” (:class:`~lenskit.data.ItemList`).  An item
list is a list of items along with additional fields, such as scores and
ratings; it supports both item IDs and numbers, and can record a vocabulary to
convert between them.  It is also backend-agnostic and can present fields
(except for the item ID) as NumPy arrays, Pandas series (optionally indexed by
item ID or number), and Torch tensors.

Item lists are used to represent a user's history, the candidate items for
scoring and/or ranking, recommendation results, test data, etc.

.. _item-list-convert:

Data Conversion
~~~~~~~~~~~~~~~

Item lists support round-tripping with Pandas :py:class:`data frames
<pandas.DataFrame>` and PyArrow :py:class:`tables <pyarrow.Table>`::

    >>> import pandas as pd
    >>> from lenskit.data import ItemList
    >>> il = ItemList(item_ids=['a', 'b'], scores=[1.5, 1.2], rating=[3, 5])
    >>> il.to_df()
        item_id     score   rating
    0         a       1.5        3
    1         b       1.2        5

.. _item-list-collections:

Item List Collections
~~~~~~~~~~~~~~~~~~~~~

On top of the :class:`~lenskit.data.ItemList` we build the idea of an item list
*collection* (`~lenskit.data.ItemListCollection`).  An item list collection is a
list or dictionary of item lists, associated with keys (e.g. the user ID).

Semantically, an :class:`ItemListCollection` is a *list* (more specifically, a
:class:`~collections.abc.Sequence`) of (key, list) tuples.  It supports the
usual sequence operations: iteration, :func:`len`, and retrieving a list and its
key by position with ``ilc[pos]``.

Lists can also be looked up by key using :meth:`~ItemListCollection.lookup`.

Keys, Schemas, and Lookup
-------------------------

.. testsetup::

    from lenskit.data import ItemListCollection, ItemList

Item list collections use **keys** following a schema that is set when the item
list collection is created.  A key schema or key type defines one or more *key
fields*, optionally with associated types, that are used to identify and look up
item lists in the collection.

In the simple, common case, item lists are associated with user IDs and the key
has a single field ``user_id``.  These keys are sufficiently common they have their
own key type, :class:`UserIDKey`.

Other experimental designs or data sets can use other key schemas.  For example, for
session-based recommendation, you may want a ``session_id`` key field.

Key schemas can be defined in two ways:

-  A named tuple type, created with either :class:`typing.NamedTuple` or
   :func:`collections.namedtuple`.
-  A list of field names.  When this method is used, LensKit creates a named
   tuple type using :func:`~collections.namedtuple` (caching it, so only a single
   type is created for each schema).

Pass the key schema to the :class:`ItemListCollection` constructor or to other
methods such as :meth:`ItemListCollection.emtpy` to create a list collection
with the specified schema::

    >>> ilc = ItemListCollection.empty(['user_id'])
    >>> ilc.key_fields
    ('user_id',)
    >>> ilc.key_type
    <class 'lenskit.data.collection._keys.UserIDKey'>

When adding to the collection, you can specify the attached key fields as
key-value pairs to the :meth:`~ItemListCollection.add` method::

    >>> ilc.add(ItemList([5, 10]), user_id=42)
    >>> len(ilc)
    1
    >>> ilc[0]
    (UserIDKey(user_id=42), <ItemList of 2 items with 0 fields {
      ids: [ 5 10]
    }>)

This list can also be retried by key with :meth:`~ItemListCollection.lookup`::

    >>> ilc.lookup(user_id=42)
    <ItemList of 2 items with 0 fields {
      ids: [ 5 10]
    }>

See the :class:`ItemListCollection` documentation for further methods.

Pandas Conversions
------------------

You can convert an item list collection from a data frame with
:meth:`~ItemListCollection.from_df`:

.. code:: python

    ilc = ItemListCollection.from_df(df, UserIDKey)

The :meth:`ItemListCollection.to_df` goes the other way, converting to a Pandas
data frame.

Saving and Loading
------------------

If you want to save or load an item list to a disk file, however, we recommend
using :meth:`~ItemListCollection.save_parquet` and
:meth:`~ItemListCollection.load_parquet` — they use a Parquet schema with one
row per list that can correctly save and load empty item lists.

Motivation
~~~~~~~~~~

Given that LensKit for Python's initial design guidelines :cite:p:`lkpy`
emphasize the use of standard data structures, why did we introduce a new
abstraction instead of continuing to use Pandas data frames?  There are a couple
of reasons for this.

*   Pandas data frames are not self-documenting; if a component returns a data
    frame, that is not enough information to know what columns to expect
    (without advanced type trickery that stretches or exceeds the limits of
    Python's typing model).  We were duplicating that knowledge across the code
    base, and things like autocomplete on the available data was not available.
    Incorrect columns was also, in Michael's experience, a common source of bugs
    and difficulties.

*   Many components were only using Pandas at the interface, and internally were
    converting to sparse matrices, tensors, etc.; by standardizing some of that
    support, and only converting data formats when necessary, we can make data
    conversions more consistent across LensKit and reduce the number of
    conversions and CPU/GPU round-trips when chaining together components using
    the same compute backend.

*   The item ID / number logic specifically was duplicated across many modules,
    and also was an early thing Michael needed to teach when teaching RecSys; a
    standard, documented abstraction that handles that logic makes it easier to
    both write components and teach recommendation concepts.
