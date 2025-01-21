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

import re
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, StringConstraints, model_validator

NAME_PATTERN = re.compile(r"^[\w_]+$")
Name: TypeAlias = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]


def id_col_name(name: str) -> str:
    return f"{name}_id"


def num_col_name(name: str) -> str:
    return f"{name}_num"


def check_name(name: str) -> None:
    """
    Check if a name is valid.

    Raises:
        ValueError: when the name is invalid.
    """
    if not NAME_PATTERN.match(name):
        raise ValueError(f"invalid name “{name}”")


class AllowableTroolean(Enum):
    """
    Three-way enumeration for storing both whether a feature is allowed and is
    used.  For convenience, in serialized data or configuration files these
    values may be specified either as strings or as booleans, in which case
    ``False`` is :attr:`FORBIDDEN` and ``True`` is :attr:`ALLOWED`.  They are
    always serialized as strings.
    """

    FORBIDDEN = "forbidden"
    """
    The feature is forbidden.
    """
    ALLOWED = "allowed"
    """
    The feature is allowed, but no records using it are present.
    """
    PRESENT = "present"
    """
    The feature is used by instances in the data.
    """

    @property
    def is_allowed(self) -> bool:
        """
        Query whether the feature is allowed.
        """
        return self != self.FORBIDDEN

    @property
    def is_forbidden(self) -> bool:
        """
        Query whether the feature is forbidden.
        """
        return self == self.FORBIDDEN

    @property
    def is_present(self) -> bool:
        """
        Query whether the feature is present (used in recorded instances).
        """
        return self == self.PRESENT

    @model_validator(mode="before")
    @classmethod
    def _validate_troolean(cls, value: Any):
        if value is True:
            return AllowableTroolean.ALLOWED
        elif value is False:
            return AllowableTroolean.FORBIDDEN
        else:
            return value


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


class DataSchema(BaseModel, extra="forbid"):
    """
    Description of the entities and layout of a dataset.
    """

    name: str | None = None
    """
    The dataset name.
    """

    default_interaction: Name | None = None
    """
    The default interaction type.
    """

    entities: dict[Name, EntitySchema] = {}
    """
    Entity classes defined for this dataset.
    """
    relationships: dict[Name, RelationshipSchema] = {}
    """
    Relationship classes defined for this dataset.
    """


class EntitySchema(BaseModel, extra="forbid"):
    """
    Entity class definitions in the dataset schema.
    """

    id_type: Literal["int", "str"] | None = None
    """
    The data type for identifiers in this entity class.
    """
    attributes: dict[Name, ColumnSpec] = {}
    """
    Entity attribute definitions.
    """


class RelationshipSchema(BaseModel, extra="forbid"):
    """
    Relationship class definitions in the dataset schema.
    """

    entities: dict[Name, Name | None]
    """
    Define the entity classes participating in the relationship.  For aliased
    entity classes (necessary for self-relationships), the key is the alias, and
    the value is the original entity class name.
    """
    interaction: bool = False
    """
    Whether this relationship class records interactions.
    """
    repeats: AllowableTroolean = AllowableTroolean.FORBIDDEN
    """
    Whether this relationship supports repeated interactions.
    """
    attributes: dict[Name, ColumnSpec] = {}
    """
    Relationship attribute definitions.
    """

    @property
    def entity_class_names(self) -> list[str]:
        return list(self.entities.keys())


class ColumnSpec(BaseModel, extra="forbid"):
    layout: AttrLayout = AttrLayout.SCALAR
    """
    The attribute layout (whether and how multiple values are supported).
    """
