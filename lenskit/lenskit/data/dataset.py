"""
LensKit dataset abstraction.
"""

# pyright: basic
from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any, Collection, Iterable, Literal, Optional, TypeAlias, TypeVar, cast, overload

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
from numpy.typing import ArrayLike, NDArray

from lenskit.data.matrix import InteractionMatrix

from .tables import NumpyUserItemTable, TorchUserItemTable

DF_FORMAT: TypeAlias = Literal["numpy", "pandas", "torch"]
MAT_FORMAT: TypeAlias = Literal["numpy", "pandas", "torch", "scipy"]
LAYOUT: TypeAlias = Literal["csr", "coo"]
ACTION_FIELDS: TypeAlias = Literal["ratings", "timestamps"] | str
EntityId: TypeAlias = Hashable
K = TypeVar("K")

_log = logging.getLogger(__name__)


class Dataset:
    """
    Representation of a data set for LensKit training, evaluation, etc. Data can
    be accessed in a variety of formats depending on the needs of a component.

    .. note::
        Zero-copy conversions are used whenever possible, so client code must not
        modify returned data in-place.

    .. todo::
        Support for item and user content or metadata is not yet implemented.
    """

    _user_counts: pd.Series[np.dtype[np.int64]]
    _item_counts: pd.Series[np.dtype[np.int64]]
    _matrix: InteractionMatrix

    def __init__(self, users: pd.Series, items: pd.Series, interact_df: pd.DataFrame):
        """
        Construct a dataset.

        .. note::
            Client code generally should not call this constructor.  Instead use the
            various ``from_`` and ``load_`` functions in :mod:`lenskit.data`.
        """
        self._user_counts = users
        self._item_counts = items
        self._init_structures(interact_df)

    def _init_structures(self, df: pd.DataFrame):
        uno = self._user_counts.index.get_indexer(df["user_id"].values)
        ino = self._item_counts.index.get_indexer(df["item_id"].values)
        assert np.all(uno >= 0)
        assert np.all(ino >= 0)

        df = df.assign(user_num=uno, item_num=ino)

        _log.debug("sorting interaction table")
        df.sort_values(["user_num", "item_num"], ignore_index=True, inplace=True)
        _log.debug("rating data frame:\n%s", df)
        if np.any(np.diff(df["item_num"]) == 0):  # pragma nocover
            raise RuntimeError("repeated ratings not yet supported")
        self._matrix = InteractionMatrix(
            uno,
            ino,
            df["rating"] if "rating" in df.columns else None,
            df["timestamp"] if "timestamp" in df.columns else None,
            self._user_counts,
            self.item_count,
        )

    @property
    def item_vocab(self) -> pd.Index:
        """
        Get the known item identifiers. This is represented as a Pandas
        :class:`~pd.Index` to enable items to be represented as contiguous item
        numbers.
        """
        return self._item_counts.index

    @property
    def user_vocab(self) -> pd.Index:
        """
        Get the known user identifiers. This is represented as a Pandas
        :class:`~pd.Index` to enable users to also be represented as contiguous
        user numbers.
        """
        return self._user_counts.index

    @property
    def item_count(self):
        return len(self.item_vocab)

    @property
    def user_count(self):
        return len(self.user_vocab)

    @overload
    def user_id(self, users: int) -> Any: ...
    @overload
    def user_id(self, users: list[int] | NDArray[np.integer]) -> pd.Series[Any]: ...
    def user_id(self, users: int | list[int] | NDArray[np.integer]) -> Any:
        """
        Look up the user ID for a given user number.  When passed a single
        number, it returns single identifier; when given an array of numbers, it
        returns a series of identifiers.

        Args:
            users: the user number(s) to look up.

        Returns:
            The user identifier(s) (from the original source data).
        """
        return _lookup_id(self.user_vocab, users)

    @overload
    def user_num(
        self, users: ArrayLike, *, missing: Literal["error", "negative"] = "negative"
    ) -> np.ndarray[int, np.dtype[np.int32]]: ...
    @overload
    def user_num(
        self, users: ArrayLike, *, missing: Literal["omit"]
    ) -> pd.Series[np.dtype[np.int32]]: ...
    @overload
    def user_num(
        self, users: Any, *, missing: Literal["error", "negative"] = "negative"
    ) -> int: ...
    def user_num(
        self, users: Any, *, missing: Literal["error", "negative", "omit"] = "negative"
    ) -> Any:
        """
        Look up the user number for a given user identifier.  When passed a
        single identifier, it returns single number; when given an array of
        numbers, it returns a series of identifiers.

        Args:
            users:
                the user identifiers(s) to look up, as used in the source data.
            missing:
                how to handle missing users (raise an error, return a negative
                value, or omit the user).  ``"omit"`` is only supported for
                arrays or lists of IDs, and returns a series index by the
                known user IDs.

        Returns:
            The user numbers.
        """
        return _lookup_num(self.user_vocab, users, missing)

    @overload
    def item_id(self, items: int) -> Any: ...
    @overload
    def item_id(self, items: list[int] | NDArray[np.integer]) -> pd.Series[Any]: ...
    def item_id(self, items: int | list[int] | NDArray[np.integer]) -> Any:
        """
        Look up the item ID for a given item number.  When passed a single
        number, it returns single identifier; when given an array of numbers, it
        returns a series of identifiers.

        Args:
            items: the item number(s) to look up.

        Returns:
            The item identifier(s) (from the original source data).
        """
        return _lookup_id(self.item_vocab, items)

    @overload
    def item_num(
        self, items: ArrayLike, *, missing: Literal["error", "negative"] = "negative"
    ) -> np.ndarray[int, np.dtype[np.int32]]: ...
    @overload
    def item_num(
        self, items: ArrayLike, *, missing: Literal["omit"]
    ) -> pd.Series[np.dtype[np.int32]]: ...
    @overload
    def item_num(
        self, items: Any, *, missing: Literal["error", "negative"] = "negative"
    ) -> int: ...
    def item_num(
        self, items: Any, *, missing: Literal["error", "negative", "omit"] = "negative"
    ) -> Any:
        """
        Look up the item number for a given item identifier.  When passed a
        single identifier, it returns single number; when given an array of
        numbers, it returns a series of identifiers.

        Args:
            items:
                the item identifiers(s) to look up, as used in the source data.
            missing:
                how to handle missing items (raise an error, return a negative
                value, or omit the item). ``"omit"`` is only supported for
                arrays or lists of IDs, and returns a series index by the
                known item IDs.

        Returns:
            The item numbers.
        """
        return _lookup_num(self.item_vocab, items, missing)

    @overload
    def interaction_log(
        self,
        format: Literal["pandas"],
        *,
        fields: str | list[str] | None = "all",
        original_ids: bool = False,
    ) -> pd.DataFrame: ...
    @overload
    def interaction_log(
        self, format: Literal["numpy"], *, fields: str | list[str] | None = "all"
    ) -> NumpyUserItemTable: ...
    @overload
    def interaction_log(
        self, format: Literal["torch"], *, fields: str | list[str] | None = "all"
    ) -> TorchUserItemTable: ...
    def interaction_log(
        self,
        format: DF_FORMAT,
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
                include ``"rating"`` and ``"timestamp"``.
            original_ids:
                If ``True``, return user and item IDs as represented in the
                original source data in columns named ``user_id`` and
                ``item_id``, instead of the user and item numbers typically
                returned.  Only applicable to the ``pandas`` format. See
                :ref:`data-identifiers`.

        Returns:
            The user-item interaction log in the specified format.
        """
        pass

    @overload
    def interaction_matrix(
        self,
        format: Literal["pandas"],
        *,
        layout: Literal["coo"] | None = None,
        field: str | None = None,
        combine: Literal["count", "sum", "mean", "first", "last"] | None = None,
        original_ids: bool = False,
    ) -> pd.DataFrame: ...
    @overload
    def interaction_matrix(
        self,
        format: Literal["torch"],
        *,
        layout: Literal["csr", "coo"] | None = None,
        field: str | None = None,
        combine: Literal["count", "sum", "mean", "first", "last"] | None = None,
    ) -> torch.Tensor: ...
    @overload
    def interaction_matrix(
        self,
        format: Literal["scipy"],
        *,
        layout: Literal["csr"] | None = None,
        legacy: bool = False,
        field: str | None = None,
        combine: Literal["count", "sum", "mean", "first", "last"] | None = None,
    ) -> sps.csr_array: ...
    @overload
    def interaction_matrix(
        self,
        format: Literal["scipy"],
        *,
        layout: Literal["csr"] | None = None,
        legacy: Literal[True],
        field: str | None = None,
        combine: Literal["count", "sum", "mean", "first", "last"] | None = None,
    ) -> sps.csr_matrix: ...
    @overload
    def interaction_matrix(
        self,
        format: Literal["scipy"],
        *,
        layout: Literal["coo"],
        legacy: bool = False,
        field: str | None = None,
        combine: Literal["count", "sum", "mean", "first", "last"] | None = None,
    ) -> sps.coo_array: ...
    @overload
    def interaction_matrix(
        self,
        format: Literal["scipy"],
        *,
        layout: Literal["coo"],
        legacy: Literal[True],
        field: str | None = None,
        combine: Literal["count", "sum", "mean", "first", "last"] | None = None,
    ) -> sps.coo_matrix: ...
    def interaction_matrix(
        self,
        format: MAT_FORMAT,
        *,
        layout: str | None = None,
        legacy: bool = False,
        field: str | None = None,
        combine: Literal["count", "sum", "mean", "first", "last"] | None = None,
        original_ids: bool = False,
    ) -> Any:
        """
        Get the user-item interactions as “ratings” matrix.  Interactions are
        not repeated.  The matrix may be in “coordinate” format, in which case
        it is comparable to :meth:`interaction_log` but without repeated
        interactions, or it may be in a compressed sparse format.

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
                * ``"scipy"`` — returns a sparse matrix from
                  :mod:`scipy.sparse`.
                * ``"structure"`` — returns a :class:`~matrix.CSRStructure`
                  containing only the user and item numbers in compressed sparse
                  row format.
            field:
                Which field to return in the matrix.  Common fields include
                ``"rating"`` and ``"timestamp"``.
            combine:
                How to combine multiple observations for a single user-item
                pair. Available methods are:

                * ``"count"`` — count the user-item interactions. Only valid
                  when ``field=None``; if the underlying data defines a
                  ``count`` field, then this is equivalent to ``"sum"`` on that
                  field.
                * ``"sum"`` — sum the field values.
                * ``"first"``, ``"last"`` — take the first or last value seen
                  (in timestamp order, if timestamps are defined).
            layout:
                The layout for a sparse matrix.  Can be either ``csr`` or
                ``coo``, or ``None`` to use the default for the specified
                format.  CSR is only supported by Torch and SciPy backends.
            original_ids:
                If ``True``, return user and item IDs as represented in the
                original source data in columns named ``user_id`` and
                ``item_id``, instead of the user and item numbers typically
                returned.  Only applicable to the ``pandas`` format. See
                :ref:`data-identifiers`.
        """
        pass


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

    Args:
        df:
            The user-item interactions (e.g. ratings).  The dataset code takes
            ownership of this data frame and may modify it.
        user_col:
            The name of the user ID column.
        item_col:
            The name of the item ID column.
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
    users = id_counts(df["user_id"])
    items = id_counts(df["item_id"])
    return Dataset(users, items, df)


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
    _log.debug("final columns: %s", kc, oc)
    return df[kc + oc]


def _find_column(columns: Collection[str], acceptable: Iterable[str]) -> str | None:
    for col in acceptable:
        if col in columns:
            return col

    return None


def id_counts(items: ArrayLike) -> pd.Series[np.dtype[np.int64]]:
    ids, counts = np.unique(items, return_counts=True)
    return pd.Series(counts, index=ids)


def _lookup_id(index: pd.Index, nums: int | list[int] | NDArray[np.integer]) -> Any:
    if np.isscalar(nums):
        nums = cast(int, nums)  # make the type checker shut up
        if nums < 0 or nums >= len(index):
            raise IndexError(f"number {nums} not in range [0,{len(index)})")
        return index[nums]
    else:
        sub = index[nums]
        return pd.Series(sub, index=nums)  # type: ignore


def _lookup_num(
    index: pd.Index,
    ids: EntityId | ArrayLike,
    missing: Literal["error", "negative", "omit"] = "negative",
) -> int | np.ndarray[int, np.dtype[np.int32]] | pd.Series[int]:
    if missing not in ["error", "negative", "omit"]:  # pragma nocover
        raise ValueError(f"invalid missing mode {missing}")
    if np.isscalar(ids):
        try:
            return index.get_loc(cast(EntityId, ids))
        except KeyError as e:
            if missing == "negative":
                return -1
            else:
                raise e
    else:
        locs = index.get_indexer(ids).astype("i4")
        if missing == "error" and np.any(locs < 0):
            raise KeyError("one or more IDs not in index")

        if missing == "omit":
            res = pd.Series(locs, index=ids)  # type: ignore
            return res[res >= 0]

        return locs
