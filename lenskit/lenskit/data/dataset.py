"""
LensKit dataset abstraction.
"""

# pyright: basic
from __future__ import annotations

from typing import Any, Literal, TypeAlias, overload

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
from numpy.typing import ArrayLike

from .tables import NumpyUserItemTable, TorchUserItemTable

DF_FORMAT: TypeAlias = Literal["numpy", "pandas", "torch"]
MAT_FORMAT: TypeAlias = Literal["numpy", "pandas", "torch", "scipy"]
LAYOUT: TypeAlias = Literal["csr", "coo"]
ACTION_FIELDS: TypeAlias = Literal["ratings", "timestamps"] | str


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

    @property
    def item_vocab(self) -> pd.Index:
        """
        Get the known item identifiers. This is represented as a Pandas
        :class:`~pd.Index` to enable items to be represented as contiguous item
        numbers.
        """
        raise NotImplementedError()

    @property
    def user_vocab(self) -> pd.Index:
        """
        Get the known user identifiers. This is represented as a Pandas
        :class:`~pd.Index` to enable users to also be represented as contiguous
        user numbers.
        """
        raise NotImplementedError()

    @overload
    def user_id(self, users: int) -> Any: ...
    @overload
    def user_id(self, users: ArrayLike) -> pd.Series[Any]: ...
    def user_id(self, users: int | ArrayLike) -> Any:
        """
        Look up the user ID for a given user number.  When passed a single
        number, it returns single identifier; when given an array of numbers, it
        returns a series of identifiers.

        Args:
            users: the user number(s) to look up.

        Returns:
            The user identifier(s) (from the original source data).
        """
        pass

    @overload
    def user_num(
        self, users: Any, *, missing: Literal["error", "negative"] = "negative"
    ) -> int: ...
    @overload
    def user_num(
        self, users: ArrayLike, *, missing: Literal["error", "negative"] = "negative"
    ) -> np.ndarray[int, np.dtype[np.int32]]: ...
    @overload
    def user_num(
        self, users: ArrayLike, *, missing: Literal["omit"]
    ) -> pd.Series[np.dtype[np.int32]]: ...
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
        pass

    @overload
    def item_id(self, items: int) -> Any: ...
    @overload
    def item_id(self, items: ArrayLike) -> pd.Series[Any]: ...
    def item_id(self, items: int | ArrayLike) -> Any:
        """
        Look up the item ID for a given item number.  When passed a single
        number, it returns single identifier; when given an array of numbers, it
        returns a series of identifiers.

        Args:
            items: the item number(s) to look up.

        Returns:
            The item identifier(s) (from the original source data).
        """
        pass

    @overload
    def item_num(
        self, items: Any, *, missing: Literal["error", "negative"] = "negative"
    ) -> int: ...
    @overload
    def item_num(
        self, items: ArrayLike, *, missing: Literal["error", "negative"] = "negative"
    ) -> np.ndarray[int, np.dtype[np.int32]]: ...
    @overload
    def item_num(
        self, items: ArrayLike, *, missing: Literal["omit"]
    ) -> pd.Series[np.dtype[np.int32]]: ...
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
        pass

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

                * ``"pandas"`` — returns a :class:`pandas.DataFrame`.
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
