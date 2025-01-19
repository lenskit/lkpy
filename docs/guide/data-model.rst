.. _data-model:

Data Model
==========

LensKit defines holistic data model for recommender training (and evaluation)
data.  The model is graph-structured, but the interfaces and definitions center
tabular (data frame) views of that data for ease of training across a variety of
statistical modeling packages.

Apache Arrow is used as the common format for data, and data type definitions
are drawn from there.  Data is transparently converted to NumPy arrays, Pandas
series or data frames, Torch tensors, etc. as requested.

Most code will either use one of the predefined dataset loading functions (such
as :func:`~lenskit.data.load_movielens`) or the
:class:`~lenskit.data.DatasetBuilder` to create data sets (see :ref:`data-api`).

.. note::

    Working with the data directly as a heterogeneous graph for integration with
    packages like PyTorch-Geometric is not difficult, and will be directly
    supported in an upcoming backwards-compatible revision.

Core Concepts
~~~~~~~~~~~~~

The LensKit data model has several core concepts, derived from
entity-relationship database model:

.. glossary::

    Entity
        The items, users, sessions, etc. about which the data set records data.
        In a graph view of the data, these are the nodes in the graph.

    Entity Class
        Each entity has a particular class, such as ``item`` or ``user``, based
        on its role in the dataset.  All data sets have at least the ``item``
        entity class.  Entities do not have subtypes in the raw data model; if
        components want to conceptually treat entities as having subtypes, such
        as different types of items, they can use attributes to distinguish the
        different subtypes.

    Entity Identifier
        Each entity has a unique (within its type) *identifier*.  Entity
        identifiers can be either integers or strings.

    Attribute
        Entities can have one or more *attributes*.  Attributes are consistent
        within an entity type, and are nullable (any individual entity may be
        missing a value for an attribute).

    Relationship
        A relationship connects two (or more) entities and may have additional
        attributes attached to the relationship itself.  Relationships may also
        be repeated (more than one relationship record may exist for the same
        combination of entities).

    Relationship Class
        Relationship classes are like entity classes, but describe the type of a
        particular relationship.  This allows for models or client code to query
        for records of a particular relationship, such as “follows” or
        “purchased”.

    Interaction
        An interaction is a specific type of relationship record that records an
        interaction between two or more entities, such as a user rating a book,
        or a user purchasing a product in a particular session.  Interactions
        usually, but not always, have timestamps.

.. _data-entities:

Entities
~~~~~~~~

*Entities* in the LensKit data model represent individual objects in the data,
such as users or items.  An entity is defined by its class and identifier, and
nothing else is directly recorded about the entity itself — the interesting data
resides in its attributes and relationships.

Entity identifiers can be integers, strings, or UUIDs.

Every data set has the entity class ``item`` for the items that may be
recommended.  Most datasets also have the class ``user``.  Session-aware
recommendation data sets may have an entity class ``session``.

When representing entities or entity data in tabular form, identifiers are
stored in a column named ``<class>_id`` (e.g. ``item_id``).  Dataset functions
that map identifiers to 0-based contiguous array indexes will use the
``<class>_num`` for this index, referred to as the *entity number*.

.. _data-attributes:

Attributes
~~~~~~~~~~

Entities (and relationships) can have associated *attributes* providing data
about that entity, relationship, or interaction.  This can be anything from a
timestamp to review text to complex item metadata.  Attributes are associated
with entity or relationship *classes*, and have types that must be consistent
across the class (each entity or relationship class has a schema defining its
attributes and their types).

Attributes come in several forms (called a *layout*):

-   **Scalar** attributes store a single value for each entity or relationship
    instance.  The value can be any type supported by NumPy or Apache Arrow.
    Attribute values may be missing, unless the schema specifies that the
    attribute is required.

-   **List** attributes store zero or more values for each entity or
    relationship instance.  List elements must have the same type.

-   **Vector** attributes store a fixed-length vector of integer or
    floating-point values for each entity or relationship instance.  The vector
    length is defined by the entity or relationship class, and must be the same
    for all instances of that class for which the vector attribute is defined.
    The vector dimensions may have associated labels or names, or they may just
    be numbered (e.g., for representing embeddings from a language model).

-   **Sparse** attributes are vector attributes that are stored in compressed
    sparse format, with missing values understood to be 0.

Attribute Name Restrictions
---------------------------

Attribute names can be freely chosen, subject to a few lightweight restrictions:

-   Within an entity or relationship class, names must be unique.
-   For each entity class ``$FOO``, the names ``$FOO_id`` and ``$FOO_num`` are
    reserved by LensKit and cannot be used by user-defined attributes (on any
    entity or relationship).  We recommend avoiding all attribute names of the
    form ``$FOO_<ident>``.

.. _data-relationships:

Relationships
~~~~~~~~~~~~~

Relationships are links between two (or more) entities, optionally with
associated attributes.  They are further divided into classes, with each class
defining its own set of relationship attributes.

Most relationships are between entities of different classes, in which case the
entity identifiers are stored in ``<class>_id`` (or ``<class>_num``) columns.
For self-relationships, however, this is not possible; such relationships must
define *aliases* for one or more of their appearances, and LensKit uses these
aliases to derive the appropriate column names.  For example, a relationship
class that encodes citation relationships in a research paper recommender system
would be a self-relationship between items.  It can alias ``item`` to ``citing``
and ``cited``, in which case the item identifiers are taken from ``citing_id``
and ``cited_id`` columns (or ``citing_num`` and ``cited_num``).

.. note::

    Entity and relationship class names must be unique (you cannot use the same
    name for an entity class and a relationship class).

.. _data-interactions:

Interactions
~~~~~~~~~~~~

An interaction is a relationship that indicates some kind of interaction between
entities for the purposes of learning and evaluating recommendations, such as
purchasing, shelving, clicking, or rating.  There is no logical difference
between relationships and interactions; an interaction class is just a
relationship class that has been declared to represent interactions, so that
client and model code knows to treat it as interaction data.  Most data sets
define a single interaction class, but can define more than one.

-   Interactions usually have timestamps (although this is not strictly
    required).  Timestamps can be either integers (treated as UNIX timestamps)
    or Arrow timestamp types.

-   The dataset can designate a *default interaction class* so that model code
    can request the “interactions” without needing to know the different classes
    involved.  If no default class is specified, and more than one class is
    defined, it is an error to request the interactions without specifying an
    interaction class.

Certain attribute names, if defined, have particular meaning for interaction
records:

``timestamp``
    The date and time of the interaction, as a UNIX or Arrow timestamp.

``rating``
    A user-supplied rating for the user-item pair.

``count``
    A count of the interactions between this pair.  If client code requests an
    matrix of interaction counts, and this attribute is defined, then its sum is
    used as the total count of interactions between the entities.  If no
    ``count`` attribute is defined, then a matrix of interaction counts is
    computed by counting the interaction records.

    .. todo::

        Define what happens when ``count`` is NULL.


.. _data-schema:

Schemas
~~~~~~~

A data *schema* (:class:`~lenskit.data.DataSchema`) defines the

.. _data-internal:

Internal Representation
~~~~~~~~~~~~~~~~~~~~~~~

Data should only be accessed through the :class:`~lenskit.data.Dataset` API, as
the internal storage is subject to change.  Logically, each entity or
relationship type is represented as a table, consisting of:

- One or more entity identifier or number columns
- Zero or more attribute columns

Data may be internally broken into sub-tables for efficiency (e.g., for very
sparse attributes), but this is the logical view. Internally, relationships use
entity numbers instead of entity IDs to record the entities involved in a
relationship record.

As of LensKit 2025.1, the native format for storing a dataset on disk (used by
:meth:`~lenskit.data.Dataset.save` and :meth:`~lenskit.data.Dataset.load`) is a
directory with a ``schema.json`` file containing the serialized logical schema
and a Parquet file ``<class>.parquet`` for each entity or relationship class
containing the identifiers and attribute values. For entity classes,
``<class>.parquet`` contains both the entity IDS and entity numbers.
