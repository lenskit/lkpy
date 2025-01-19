# pyright: strict
from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetManifest(BaseModel):
    """
    Description of the entities and layout of a dataset.
    """

    entities: dict[str, str | None] = Field(default_factory=dict)
