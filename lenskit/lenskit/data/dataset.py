# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit dataset abstraction.
"""

# pyright: basic
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch

from .items import ItemList
from .tables import NumpyUserItemTable, TorchUserItemTable
from .types import ID
from .vocab import Vocabulary

DF_FORMAT: TypeAlias = Literal["numpy", "pandas", "torch"]
MAT_FORMAT: TypeAlias = Literal["scipy", "torch", "pandas", "structure"]
MAT_AGG: TypeAlias = Literal["count", "sum", "mean", "first", "last"]
LAYOUT: TypeAlias = Literal["csr", "coo"]
ACTION_FIELDS: TypeAlias = Literal["ratings", "timestamps"] | str

K = TypeVar("K")


class FieldError(KeyError):
    """
    The requested field does not exist.
    """

    def __init__(self, entity, field):
        super().__init__(f"{entity}[{field}]")


class Dataset(ABC):
    """
    Representation of a data set for LensKit training, evaluation, etc. Data can
    be accessed in a variety of formats depending on the needs of a component.

    .. note::
        Zero-copy conversions are used whenever possible, so client code must not
        modify returned data in-place.

    .. todo::
        Support for advanced rating situations is not yet supported:

        * repeated ratings
        * mixed implicit & explicit feedback
        * later actions removing earlier ratings

    .. todo::
        Support for item and user content or metadata is not yet implemented.

    .. stability:: caller
    """

    _item_stats: pd.DataFrame | None = None
    _user_stats: pd.DataFrame | None = None

    @property
    @abstractmethod
    def items(self) -> Vocabulary:
        """
        The items known by this dataset.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def users(self) -> Vocabulary:
        """
        The users known by this dataset.
        """
        raise NotImplementedError()

    @abstractmethod
    def count(self, what: str) -> int:
        """
        Count entities in the dataset.

        .. note::

            The precise counts are subtle in the presence of repeated or
            superseded interactions. See :meth:`interaction_count` and
            :meth:`rating_count` for details on the ``"interactions"`` and
            ``"ratings"`` counts.

        Args:
            what:
                The type of entity to count.  Commonly-supported ones include:

                * users
                * items
                * pairs (observed user-item pairs)
                * interactions
                * ratings
        """
        raise NotImplementedError()

    @property
    def item_count(self) -> int:
        return self.count("items")

    @property
    def user_count(self) -> int:
        return self.count("users")

    @property
    def interaction_count(self) -> int:
        """
        Count the total number of interaction records.  Equivalent to
        ``count("interactions")``.

        .. note::
            If the interaction records themselves reprsent counts, such as the
            number of times a song was played, this returns the number of
            *records*, not the total number of plays.
        """
        return self.count("interactions")

    @property
    def rating_count(self) -> int:
        """
        Count the total number of ratings (excluding superseded ratings).
        Equivalent to ``count("ratings")``.
        """
        return self.count("ratings")

    @overload
    @abstractmethod
    def interaction_log(
        self,
        format: Literal["pandas"],
        *,
        fields: str | list[str] | None = "all",
        original_ids: bool = False,
    ) -> pd.DataFrame: ...
    @overload
    @abstractmethod
    def interaction_log(
        self, format: Literal["numpy"], *, fields: str | list[str] | None = "all"
    ) -> NumpyUserItemTable: ...
    @overload
    @abstractmethod
    def interaction_log(
        self, format: Literal["torch"], *, fields: str | list[str] | None = "all"
    ) -> TorchUserItemTable: ...
    @abstractmethod
    def interaction_log(
        self,
        format: str,
        *,
        fields: str | list[str] | None = "all",
        original_ids: bool = False,
    ) -> Any:
        """
        Get the user-item interactions as a table in the requested format. The
        table is not in a specified order.  Interactions may be repeated (e.g.
        the same user may listen to a song multiple times).  For a non-repeated
        “ratings matrix” view of the data, see :meth:`interaction_matrix`.

        .. warning::
            Client code **must not** perform in-place modifications on the table
            returned from this method.  Whenever possible, it will be a shallow
            view on top of the underlying storage, and modifications may corrupt
            data for other code.

        Args:
            format:
                The desired data format.  Currently-supported formats are:

                * ``"pandas"`` — returns a :class:`pandas.DataFrame`.  The index
                  is not meaningful.
                * ``"numpy"`` — returns a :class:`~tables.NumpyUserItemTable`.
                * ``"torch"`` — returns a :class:`~tables.TorchUserItemTable`.
            fields:
                Which fields to include.  If set to ``"all"``, will include all
                available fields in the resulting table; ``None`` includes no
                fields besides the user and item.  Commonly-available fields
                include ``"rating"`` and ``"timestamp"``.  Missing fields will
                be omitted in the result.
            original_ids:
                If ``True``, return user and item IDs as represented in the
                original source data in columns named ``user_id`` and
                ``item_id``, instead of the user and item numbers typically
                returned.  Only applicable to the ``pandas`` format. See
                :ref:`data-identifiers`.

        Returns:
            The user-item interaction log in the specified format.
        """
        raise NotImplementedError()

    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        format: Literal["pandas"],
        *,
        layout: Literal["coo"] | None = None,
        field: str | None = None,
        combine: MAT_AGG | None = None,
        original_ids: bool = False,
    ) -> pd.DataFrame: ...
    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        format: Literal["torch"],
        *,
        layout: Literal["csr", "coo"] | None = None,
        field: str | None = None,
        combine: MAT_AGG | None = None,
    ) -> torch.Tensor: ...
    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        format: Literal["scipy"],
        *,
        layout: Literal["coo"],
        legacy: Literal[True],
        field: str | None = None,
        combine: MAT_AGG | None = None,
    ) -> sps.coo_matrix: ...
    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        format: Literal["scipy"],
        *,
        layout: Literal["coo"],
        legacy: bool = False,
        field: str | None = None,
        combine: MAT_AGG | None = None,
    ) -> sps.coo_array: ...
    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        format: Literal["scipy"],
        *,
        layout: Literal["csr"] | None = None,
        legacy: Literal[True],
        field: str | None = None,
        combine: MAT_AGG | None = None,
    ) -> sps.csr_matrix: ...
    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        format: Literal["scipy"],
        *,
        layout: Literal["csr"] | None = None,
        legacy: bool = False,
        field: str | None = None,
        combine: MAT_AGG | None = None,
    ) -> sps.csr_array: ...
    @overload
    @abstractmethod
    def interaction_matrix(
        self,
        format: Literal["structure"],
        *,
        layout: Literal["csr"] | None = None,
    ) -> CSRStructure: ...
    @abstractmethod
    def interaction_matrix(
        self,
        format: str,
        *,
        layout: str | None = None,
        legacy: bool = False,
        field: str | None = None,
        combine: str | None = None,
        original_ids: bool = False,
    ) -> Any:
        """
        Get the user-item interactions as “ratings” matrix.  Interactions are
        not repeated.  The matrix may be in “coordinate” format, in which case
        it is comparable to :meth:`interaction_log` but without repeated
        interactions, or it may be in a compressed sparse format.

        .. todo::
            Aggregate is currently ignored because repeated interactions are not
            yet supported.

        .. warning::
            Client code **must not** perform in-place modifications on the matrix
            returned from this method.  Whenever possible, it will be a shallow
            view on top of the underlying storage, and modifications may corrupt
            data for other code.

        Args:
            format:
                The desired data format.  Currently-supported formats are:

                * ``"pandas"`` — returns a :class:`pandas.DataFrame`.
                * ``"torch"`` — returns a sparse :class:`torch.Tensor` (see
                  :mod:`torch.sparse`).
                * ``"scipy"`` — returns a sparse array from :mod:`scipy.sparse`.
                * ``"structure"`` — returns a :class:`~matrix.CSRStructure`
                  containing only the user and item numbers in compressed sparse
                  row format.
            field:
                Which field to return in the matrix.  Common fields include
                ``"rating"`` and ``"timestamp"``.

                If unspecified (``None``), this will yield an implicit-feedback
                indicator matrix, with 1s for observed items; the ``"pandas"``
                format will only include user and item columns.

                If the ``rating`` field is requested but is not defined in the
                underlying data, then this is equivalent to ``"indicator"``,
                except that the ``"pandas"`` format will include a ``"rating"``
                column of all 1s.

                The ``"pandas"`` format also supports the special field name
                ``"all"`` to return a data frame with all available fields. When
                ``field="all"``, a field named ``count`` (if defined) is
                combined with the ``sum`` method, and other fields use ``last``.
            combine:
                How to combine multiple observations for a single user-item
                pair. Available methods are:

                * ``"count"`` — count the user-item interactions. Only valid
                  when ``field=None``; if the underlying data defines a
                  ``count`` field, then this is equivalent to ``"sum"`` on that
                  field.
                * ``"sum"`` — sum the field values.
                * ``"first"``, ``"last"`` — take the first or last value seen
                  (in timestamp order, if timestamps are defined; otherwise,
                  their order in the original input).
            layout:
                The layout for a sparse matrix.  Can be either ``csr`` or
                ``coo``, or ``None`` to use the default for the specified
                format.  CSR is only supported by Torch and SciPy backends.
            legacy:
                ``True`` to return a legacy SciPy sparse matrix instead of
                sparse array.
            original_ids:
                ``True`` to return user and item IDs instead of numbers in
                ``pandas``-format matrix.
        """
        raise NotImplementedError()

    @abstractmethod
    @overload
    def user_row(self, user_id: ID) -> ItemList | None: ...
    @abstractmethod
    @overload
    def user_row(self, *, user_num: int) -> ItemList: ...
    @abstractmethod
    def user_row(
        self, user_id: ID | None = None, *, user_num: int | None = None
    ) -> ItemList | None:
        """
        Get a user's row from the interaction matrix.  Available fields are
        returned as fields. If the dataset has ratings, these are provided as a
        ``rating`` field, **not** as the item scores.  The item list is
        unordered, but items are returned in order by item number.

        Args:
            user_id:
                The ID of the user to retrieve.
            user_num:
                The number of the user to retrieve.

        Returns:
            The user's interaction matrix row, or ``None`` if no user with that ID
            exists.
        """
        raise NotImplementedError()

    def item_stats(self) -> pd.DataFrame:
        """
        Get item statistics.

        Returns:
            A data frame indexed by item ID with the following columns:

            * count — the number of interactions recorded for this item.
            * user_count — the number of distinct users who have interacted with
              or rated this item.
            * rating_count — the number of ratings for this item.  Only provided
              if the dataset has explicit ratings; if there are repeated
              ratings, this does **not** count superseded ratings.
            * mean_rating — the mean of the ratings. Only provided if the
              dataset has explicit ratings.
            * first_time — the first time the item appears. Only provided if the
              dataset has timestamps.

            The index is the vocabulary, so ``iloc`` works with item numbers.
        """

        if self._item_stats is None:
            log = self.interaction_log("numpy")

            counts = np.zeros(self.item_count, dtype=np.int32)
            np.add.at(counts, log.item_nums, 1)
            frame = pd.DataFrame(
                {
                    "count": counts,
                    "user_count": counts,
                },
                index=self.items.index,
            )

            if log.ratings is not None:
                sums = np.zeros(self.item_count, dtype=np.float64)
                np.add.at(sums, log.item_nums, log.ratings)
                frame["rating_count"] = counts
                frame["mean_rating"] = sums / counts

            if log.timestamps is not None:
                i64i = np.iinfo(np.int64)
                times = np.full(self.item_count, i64i.max, dtype=np.int64)
                np.minimum.at(times, log.item_nums, log.timestamps)
                frame["first_time"] = times

            self._item_stats = frame

        return self._item_stats

    def user_stats(self) -> pd.DataFrame:
        """
        Get user statistics.

        Returns:
            A data frame indexed by user ID with the following columns:

            * count — the number of interactions recorded for this user.
            * item_count — the number of distinct items with which this user has
              interacted.
            * rating_count — the number of ratings for this user.  Only provided
              if the dataset has explicit ratings; if there are repeated
              ratings, this does **not** count superseded ratings.
            * mean_rating — the mean of the user's reatings. Only provided if
              the dataset has explicit ratings.
            * first_time — the first time the user appears. Only provided if the
              dataset has timestamps.
            * last_time — the last time the user appears. Only provided if the
              dataset has timestamps.

            The index is the vocabulary, so ``iloc`` works with user numbers.
        """

        if self._user_stats is None:
            log = self.interaction_log("numpy")

            counts = np.zeros(self.user_count, dtype=np.int32)
            np.add.at(counts, log.user_nums, 1)
            frame = pd.DataFrame(
                {
                    "count": counts,
                    "item_count": counts,
                },
                index=self.users.index,
            )

            if log.ratings is not None:
                sums = np.zeros(self.user_count, dtype=np.float64)
                np.add.at(sums, log.user_nums, log.ratings)
                frame["rating_count"] = counts
                frame["mean_rating"] = sums / counts

            if log.timestamps is not None:
                i64i = np.iinfo(np.int64)
                first = np.full(self.user_count, i64i.max, dtype=np.int64)
                last = np.full(self.user_count, i64i.min, dtype=np.int64)
                np.minimum.at(first, log.user_nums, log.timestamps)
                np.maximum.at(last, log.user_nums, log.timestamps)
                frame["first_time"] = first
                frame["last_time"] = last

            self._user_stats = frame

        return self._user_stats


from .matrix import CSRStructure  # noqa: E402
