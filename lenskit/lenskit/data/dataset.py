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

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
from numpy.typing import ArrayLike
from typing_extensions import (
    Any,
    Callable,
    Collection,
    Iterable,
    Literal,
    Optional,
    TypeAlias,
    TypeVar,
    overload,
    override,
)

from lenskit.types import EntityId

from .items import ItemList
from .matrix import CSRStructure, InteractionMatrix
from .tables import NumpyUserItemTable, TorchUserItemTable
from .vocab import Vocabulary

DF_FORMAT: TypeAlias = Literal["numpy", "pandas", "torch"]
MAT_FORMAT: TypeAlias = Literal["scipy", "torch", "pandas", "structure"]
MAT_AGG: TypeAlias = Literal["count", "sum", "mean", "first", "last"]
LAYOUT: TypeAlias = Literal["csr", "coo"]
ACTION_FIELDS: TypeAlias = Literal["ratings", "timestamps"] | str

K = TypeVar("K")

_log = logging.getLogger(__name__)


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
    def user_row(self, user_id: EntityId) -> ItemList | None: ...
    @abstractmethod
    @overload
    def user_row(self, *, user_num: int) -> ItemList: ...
    @abstractmethod
    def user_row(
        self, user_id: EntityId | None = None, *, user_num: int | None = None
    ) -> ItemList | None:
        """
        Get a user's row from the interaction matrix.  Available fields are
        returned as fields. If the dataset has ratings, these are provided as a
        ``rating`` field, **not** as the item scores.  The item list is unordered,
        but items are returned in order by item number.
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
            * mean_rating — the mean of the reatings. Only provided if the
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
                    "user_count": counts,
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


class MatrixDataset(Dataset):
    """
    Dataset implementation using an in-memory rating or implicit-feedback matrix
    (with no duplicate interactions).

    .. note::
        Client code generally should not construct this class directly.  Instead
        use the various ``from_`` and ``load_`` functions in
        :mod:`lenskit.data`.
    """

    _users: Vocabulary
    "User ID vocabulary, to map between IDs and row numbers."
    _items: Vocabulary
    "Item ID vocabulary, to map between IDs and column or row numbers."
    _matrix: InteractionMatrix

    def __init__(self, users: Vocabulary, items: Vocabulary, interact_df: pd.DataFrame):
        """
        Construct a dataset.

        .. note::
            Client code generally should not call this constructor.  Instead use the
            various ``from_`` and ``load_`` functions in :mod:`lenskit.data`.
        """
        self._users = users
        self._items = items
        self._init_structures(interact_df)

    def _init_structures(self, df: pd.DataFrame):
        uno = self.users.numbers(df["user_id"])
        ino = self.items.numbers(df["item_id"])
        assert np.all(uno >= 0)
        assert np.all(ino >= 0)
        if np.any(df.duplicated(subset=["user_id", "item_id"])):
            raise RuntimeError("repeated ratings not yet supported")

        df = df.assign(user_num=uno, item_num=ino)

        _log.debug("sorting interaction table")
        df.sort_values(["user_num", "item_num"], ignore_index=True, inplace=True)
        _log.debug("rating data frame:\n%s", df)
        self._matrix = InteractionMatrix(
            uno,
            ino,
            df["rating"] if "rating" in df.columns else None,
            df["timestamp"] if "timestamp" in df.columns else None,
            self.user_count,
            self.item_count,
        )

    @property
    @override
    def items(self) -> Vocabulary:
        return self._items

    @property
    @override
    def users(self) -> Vocabulary:
        return self._users

    @override
    def count(self, what: str) -> int:
        match what:
            case "users":
                return self._users.size
            case "items":
                return self._items.size
            case "pairs" | "interactions" | "ratings":
                return self._matrix.n_obs
            case _:
                raise KeyError(f"unknown entity type {what}")

    @override
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
        match format:
            case "structure":
                if layout and layout != "csr":
                    raise ValueError(f"unsupported layout {layout} for structure")
                if field:
                    raise ValueError("structure does not support fields")
                return self._int_mat_structure()
            case "pandas":
                if layout and layout != "coo":
                    raise ValueError(f"unsupported layout {layout} for Pandas")
                return self._int_mat_pandas(field, original_ids)
            case "scipy":
                return self._int_mat_scipy(field, layout, legacy)
            case "torch":
                return self._int_mat_torch(field, layout)
            case _:
                raise ValueError(f"unsupported format “{format}”")

    def _int_mat_structure(self) -> CSRStructure:
        return CSRStructure(self._matrix.user_ptrs, self._matrix.item_nums, self._matrix.shape)

    def _int_mat_pandas(self, field: str | None, original_ids: bool) -> pd.DataFrame:
        cols: dict[str, ArrayLike]
        if original_ids:
            cols = {
                "user_id": self.users.ids(self._matrix.user_nums),
                "item_id": self.items.ids(self._matrix.item_nums),
            }
        else:
            cols = {
                "user_num": self._matrix.user_nums,
                "item_num": self._matrix.item_nums,
            }
        if field == "all" or field == "rating":
            if self._matrix.ratings is not None:
                cols["rating"] = self._matrix.ratings
            else:
                cols["rating"] = np.ones(self._matrix.n_obs)
        elif field == "all" or field == "timestamp":
            if self._matrix.timestamps is None:
                raise FieldError("interaction", field)
            cols["timestamp"] = self._matrix.timestamps
        elif field and field != "all":
            raise FieldError("interaction", field)
        return pd.DataFrame(cols)

    def _int_mat_scipy(self, field: str | None, layout: str | None, legacy: bool):
        if field == "rating" and self._matrix.ratings is not None:
            data = self._matrix.ratings
        elif field is None or field == "rating":
            data = np.ones(self._matrix.n_obs, dtype="f4")
        elif field == "timestamp" and self._matrix.timestamps is not None:
            data = self._matrix.timestamps
        else:  # pragma nocover
            raise FieldError("interaction", field)

        shape = self._matrix.shape

        if layout is None:
            layout = "csr"
        match layout:
            case "csr":
                ctor = sps.csr_matrix if legacy else sps.csr_array
                return ctor((data, self._matrix.item_nums, self._matrix.user_ptrs), shape=shape)
            case "coo":
                ctor = sps.coo_matrix if legacy else sps.coo_array
                return ctor((data, (self._matrix.user_nums, self._matrix.item_nums)), shape=shape)
            case _:  # pragma nocover
                raise ValueError(f"unsupported layout {layout}")

    def _int_mat_torch(self, field: str | None, layout: str | None):
        if field == "rating" and self._matrix.ratings is not None:
            values = torch.from_numpy(self._matrix.ratings)
        elif field is None or field == "rating":
            values = torch.full([self._matrix.n_obs], 1.0, dtype=torch.float32)
        elif field == "timestamp" and self._matrix.timestamps is not None:
            values = torch.from_numpy(self._matrix.timestamps)
        else:  # pragma nocover
            raise FieldError("interaction", field)

        shape = self._matrix.shape

        if layout is None:
            layout = "csr"
        match layout:
            case "csr":
                return torch.sparse_csr_tensor(
                    torch.from_numpy(self._matrix.user_ptrs),
                    torch.from_numpy(self._matrix.item_nums),
                    values,
                    size=shape,
                )
            case "coo":
                indices = np.stack([self._matrix.user_nums, self._matrix.item_nums], dtype=np.int32)
                return torch.sparse_coo_tensor(
                    torch.from_numpy(indices),
                    values,
                    size=shape,
                ).coalesce()
            case _:  # pragma nocover
                raise ValueError(f"unsupported layout {layout}")

    @override
    def interaction_log(
        self,
        format: str,
        *,
        fields: str | list[str] | None = "all",
        original_ids: bool = False,
    ) -> Any:
        if fields == "all":
            fields = ["rating", "timestamp"]
        elif isinstance(fields, str):
            fields = [fields]
        elif fields is None:
            fields = []

        match format:
            case "pandas":
                return self._int_log_pandas(fields, original_ids)
            case "numpy":
                return self._int_log_numpy(fields)
            case "torch":
                return self._int_log_torch(fields)
            case _:
                raise ValueError(f"unsupported format “{format}”")

    def _int_log_pandas(self, fields: list[str], original_ids: bool):
        cols: dict[str, ArrayLike]
        if original_ids:
            cols = {
                "user_id": self.users.terms(self._matrix.user_nums),
                "item_id": self.items.terms(self._matrix.item_nums),
            }
        else:
            cols = {
                "user_num": self._matrix.user_nums,
                "item_num": self._matrix.item_nums,
            }
        if "rating" in fields and self._matrix.ratings is not None:
            cols["rating"] = self._matrix.ratings
        if "timestamp" in fields and self._matrix.timestamps is not None:
            cols["timestamp"] = self._matrix.timestamps
        return pd.DataFrame(cols)

    def _int_log_numpy(self, fields: list[str]) -> NumpyUserItemTable:
        tbl = NumpyUserItemTable(self._matrix.user_nums, self._matrix.item_nums)
        if "rating" in fields:
            tbl.ratings = self._matrix.ratings
        if "timestamp" in fields:
            tbl.timestamps = self._matrix.timestamps
        return tbl

    def _int_log_torch(self, fields: list[str]) -> TorchUserItemTable:
        tbl = TorchUserItemTable(
            torch.from_numpy(self._matrix.user_nums), torch.from_numpy(self._matrix.item_nums)
        )
        if "rating" in fields:
            tbl.ratings = torch.from_numpy(self._matrix.ratings)
        if "timestamp" in fields:
            tbl.timestamps = torch.from_numpy(self._matrix.timestamps)
        return tbl

    @override
    def user_row(
        self, user_id: EntityId | None = None, *, user_num: int | None = None
    ) -> ItemList | None:
        if user_num is None:
            if user_id is None:  # pragma: nocover
                raise ValueError("most provide one of user_id and user_num")

            user_num = self.users.number(user_id, "none")
            if user_num is None:
                return None

        elif user_id is not None:  # pragma: nocover
            raise ValueError("most provide one of user_id and user_num")

        sp = self._matrix.user_ptrs[user_num]
        ep = self._matrix.user_ptrs[user_num + 1]
        inums = self._matrix.item_nums[sp:ep]
        fields = {}
        if self._matrix.ratings is not None:
            fields["rating"] = self._matrix.ratings[sp:ep]
        if self._matrix.timestamps is not None:
            fields["timestamp"] = self._matrix.timestamps[sp:ep]
        return ItemList(item_nums=inums, vocabulary=self.items, **fields)


class LazyDataset(Dataset):
    """
    A data set with an underlying load function, that doesn't call the function
    until data is actually needed.

    Args:
        loader:
            The function that will load the dataset when needed.
    """

    _delegate: Dataset | None = None
    _loader: Callable[[], Dataset]

    def __init__(self, loader: Callable[[], Dataset]):
        """
        Construct a lazy dataset.
        """
        self._loader = loader

    def delegate(self) -> Dataset:
        """
        Get the delegate data set, loading it if necessary.
        """
        if self._delegate is None:
            self._delegate = self._loader()
        return self._delegate

    @property
    @override
    def items(self) -> Vocabulary:
        return self.delegate().items

    @property
    @override
    def users(self) -> Vocabulary:
        return self.delegate().users

    @override
    def count(self, what: str) -> int:
        return self.delegate().count(what)

    @override
    def interaction_matrix(self, *args, **kwargs) -> Any:
        return self.delegate().interaction_matrix(*args, **kwargs)

    @override
    def interaction_log(self, *args, **kwargs) -> Any:
        return self.delegate().interaction_log(*args, **kwargs)

    @override
    def user_row(self, *args, **kwargs) -> ItemList | None:
        return self.delegate().user_row(*args, **kwargs)


def from_interactions_df(
    df: pd.DataFrame,
    *,
    user_col: Optional[str] = None,
    item_col: Optional[str] = None,
    rating_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> Dataset:
    """
    Create a dataset from a data frame of ratings or other user-item
    interactions.

    .. todo::
        Repeated interactions are not yet supported.

    Args:
        df:
            The user-item interactions (e.g. ratings).  The dataset code takes
            ownership of this data frame and may modify it.
        user_col:
            The name of the user ID column.  By default, looks for columns named
            ``user``, ``user_id``, or ``userId``, with several case variants.
        item_col:
            The name of the item ID column.  By default, looks for columns named
            ``item``, ``item_id``, or ``itemId``, with several case variants.
        rating_col:
            The name of the rating column.
        timestamp_col:
            The name of the timestamp column.
    """
    _log.info("creating data set from %d x %d data frame", len(df.columns), len(df))
    df = normalize_interactions_df(
        df,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        timestamp_col=timestamp_col,
    )
    df = df.sort_values(["user_id", "item_id"])
    users = Vocabulary(df["user_id"], "user")
    items = Vocabulary(df["item_id"], "item")
    return MatrixDataset(users, items, df)


def normalize_interactions_df(
    df: pd.DataFrame,
    *,
    user_col: Optional[str] = None,
    item_col: Optional[str] = None,
    rating_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Normalize the column names and layout for an interaction data frame.
    """
    _log.debug("normalizing data frame with columns %s", df.columns)
    if user_col is None:
        user_col = _find_column(
            df.columns,
            ["user_id", "user", "USER", "userId", "UserId"],
        )
    if user_col is None:  # pragma nocover
        raise ValueError("no user column found")
    if item_col is None:
        item_col = _find_column(
            df.columns,
            ["item_id", "item", "ITEM", "itemId", "ItemId"],
        )
    if item_col is None:  # pragma nocover
        raise ValueError("no item column found")
    if rating_col is None:
        rating_col = _find_column(
            df.columns,
            ["rating", "RATING"],
        )
    if timestamp_col is None:
        timestamp_col = _find_column(
            df.columns,
            ["timestamp", "TIMESTAMP"],
        )

    _log.debug("id columns: user=%s, item=%s", user_col, item_col)
    _log.debug("rating column: %s", rating_col)
    _log.debug("timestamp column: %s", timestamp_col)

    # rename and reorder columns
    known_columns = ["user_id", "item_id", "rating", "timestamp", "count"]
    renames = {user_col: "user_id", item_col: "item_id"}
    if rating_col:
        renames[rating_col] = "rating"
    if timestamp_col:
        renames[timestamp_col] = "timestamp"
    df = df.rename(columns=renames)
    kc = [c for c in known_columns if c in df.columns]
    oc = [c for c in df.columns if c not in known_columns]
    _log.debug("final columns: %s", kc + oc)
    return df[kc + oc]  # type: ignore


def _find_column(columns: Collection[str], acceptable: Iterable[str]) -> str | None:
    for col in acceptable:
        if col in columns:
            return col

    return None
