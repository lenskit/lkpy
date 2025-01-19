"""
Pydantic models for LensKit data schemas.  These models define define the data
schema in memory and also define how schemas are serialized to and from
configuration files.  See :ref:`data-model` for details.

.. note::

    The schema does not specify data types directly — data types are inferred
    from the underlying Arrow data structures.  This reduces duplication of type
    information and the opportunity for inconsistency.
"""

# pyright: strict
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class AttrLayout(Enum):
    SCALAR = "scalar"
    """
    Scalar (non-list, non-vector) attribute value.
    """
    LIST = "list"
    """
    Homogenous, variable-length list of attribute values.
    """
    VECTOR = "vector"
    """
    Homogenous, fixed-length vector of numeric attribute values.
    """
    SPARSE = "sparse"
    """
    Homogenous, fixed-length sparse vector of numeric attribute values.
    """


class DataSchema(BaseModel):
    """
    Description of the entities and layout of a dataset.
    """

    entities: dict[str, EntitySchema] = {}
    """
    Entity classes defined for this dataset.
    """
    relationships: dict[str, RelationshipSchema] = {}
    """
    Relationship classes defined for this dataset.
    """


class EntitySchema(BaseModel):
    """
    Entity class definitions in the dataset schema.
    """

    id_type: Literal["int", "str", "uuid"] | None = None
    """
    The data type for identifiers in this entity class.
    """
    attributes: list[str] | dict[str, ColumnSpec] = {}
    """
    Entity attribute definitions.
    """


class RelationshipSchema(BaseModel):
    """
    Relationship class definitions in the dataset schema.
    """

    entities: list[str] | dict[str, str | None]
    """
    Define the entities participating in the relationship.  If specified as a
    mapping, defines the aliases for the entities used to determine their column
    names (necessary for self-relationships).
    """
    interaction: bool = False
    """
    Whether this relationship class records interactions.
    """
    attributes: list[str] | dict[str, ColumnSpec] = {}
    """
    Relationship attribute definitions.
    """


class ColumnSpec(BaseModel):
    required: bool = False
    """
    Whether the attribute is required to have a value.
    """
    layout: AttrLayout = AttrLayout.SCALAR
    """
    The attribute layout (whether and how multiple values are supported).
    """
