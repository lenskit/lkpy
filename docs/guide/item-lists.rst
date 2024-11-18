Item Lists and Collections
==========================

Throughout its data handling, components, and evaluation metrics, LensKit uses
an abstraction of an “item list” (:class:`~lenskit.data.ItemList`).  An item
list is a list of items along with additional fields, such as scores and
ratings; it supports both item IDs and numbers, and can record a vocabulary to
convert between them.  It is also backend-agnostic and can present fields
(except for the item ID) as NumPy arrays, Pandas series (optionally indexed by
item ID or number), and Torch tensors.

Item lists are used to represent a user's history, the candidate items for
scoring and/or ranking, recommendation results, test data, etc.

.. todo::

    Write more tutorial / user manual documentation for the item lists.

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

Collections
~~~~~~~~~~~

On top of the :class:`~lenskit.data.ItemList` we build the idea of an item list
*collection* (`~lenskit.data.ItemListCollection`).  An item list collection is a
list or dictionary of item lists, identified by keys (e.g. the user ID )

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
