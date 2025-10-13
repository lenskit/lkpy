# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Mixins with commonly-used component configuration capabilities.
"""

from __future__ import annotations

from pydantic import PositiveInt, model_validator


class EmbeddingSizeMixin:
    """
    Mixin for configuring embedding sizes (# of latent dimensions).

    Component configuration classes can extend this class to inherit a
    standardized definition of an embedding size, along with useful behavior
    like configuring with base-2 logs.

    Example usage:

    .. code:: python

        class SVDConfig(EmbeddingSizeMixin, BaseModel):
            pass

        cfg = SVDConfig(embedding_size=32)
    """

    embedding_size: PositiveInt
    """
    The dimension of user and item embeddings (number of latent features to
    learn).
    """

    @model_validator(mode="before")
    @classmethod
    def lkmv_embedding_size(cls, data):
        match data:
            case {"embedding_size_exp": e}:
                # convert embedding size and remove from data
                return {"embedding_size": 2**e} | {
                    k: v
                    for k, v in data.items()
                    if k not in ("embedding_size_exp", "embedding_size")
                }
            case _:
                return data
