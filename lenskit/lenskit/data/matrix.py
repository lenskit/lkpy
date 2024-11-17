# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data manipulation routines.
"""

# pyright: basic
from __future__ import annotations

import logging
from typing import Any, NamedTuple, Optional, TypeVar

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
from numpy.typing import ArrayLike
from typing_extensions import override

from .dataset import Dataset, FieldError
from .items import ItemList
from .tables import NumpyUserItemTable, TorchUserItemTable
from .types import ID
from .vocab import Vocabulary

_log = logging.getLogger(__name__)

t = torch
M = TypeVar("M", "CSRStructure", sps.csr_array, sps.coo_array, sps.spmatrix, t.Tensor)


class CSRStructure(NamedTuple):
    """
    Representation of the compressed sparse row structure of a sparse matrix,
    without any data values.
    """

    rowptrs: np.ndarray
    colinds: np.ndarray
    shape: tuple[int, int]

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    @property
    def nnz(self):
        return self.rowptrs[self.nrows]

    def extent(self, row: int) -> tuple[int, int]:
        return self.rowptrs[row], self.rowptrs[row + 1]

    def row_cs(self, row: int) -> np.ndarray:
        sp, ep = self.extent(row)
        return self.colinds[sp:ep]


class InteractionMatrix:
    """
    Internal helper class used by :class:`lenskit.data.Dataset` to store the
    user-item interaction matrix.  The data is stored simultaneously in CSR and
    COO format.  Most code has no need to interact with this class directly —
    :class:`~lenskit.data.Dataset` methods provide data in a range of formats.
    """

    n_obs: int
    n_users: int
    n_items: int

    user_nums: np.ndarray[int, np.dtype[np.int32]]
    "User (row) numbers."
    user_ptrs: np.ndarray[int, np.dtype[np.int32]]
    "User (row) offsets / pointers."
    item_nums: np.ndarray[int, np.dtype[np.int32]]
    "Item (column) numbers."
    ratings: Optional[np.ndarray[int, np.dtype[np.float32]]] = None
    "Rating values."
    timestamps: Optional[np.ndarray[int, np.dtype[np.int64]]] = None
    "Timestamps as 64-bit Unix timestamps."

    def __init__(
        self,
        users: ArrayLike,
        items: ArrayLike,
        ratings: Optional[ArrayLike],
        timestamps: Optional[ArrayLike],
        n_users: int,
        n_items: int,
    ):
        self.user_nums = np.asarray(users, np.int32)
        assert np.all(np.diff(self.user_nums) >= 0), "matrix data not sorted"
        self.item_nums = np.asarray(items, np.int32)
        if ratings is not None:
            self.ratings = np.asarray(ratings, np.float32)
        if timestamps is not None:
            self.timestamps = np.asarray(timestamps, np.int64)

        self.n_obs = len(self.user_nums)
        self.n_items = n_items
        self.n_users = n_users
        cp1 = np.zeros(self.n_users + 1, np.int32)
        np.add.at(cp1[1:], self.user_nums, 1)
        self.user_ptrs = cp1.cumsum(dtype=np.int32)
        if self.user_ptrs[-1] != len(self.user_nums):
            raise ValueError("mismatched counts and array sizes")

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the interaction matrix (rows x columns).
        """
        return (self.n_users, self.n_items)


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
        self, user_id: ID | None = None, *, user_num: int | None = None
    ) -> ItemList | None:
        if user_num is None:
            if user_id is None:  # pragma: nocover
                raise ValueError("must provide one of user_id and user_num")

            user_num = self.users.number(user_id, "none")
            if user_num is None:
                return None

        elif user_id is not None:  # pragma: nocover
            raise ValueError("must provide one of user_id and user_num")

        sp = self._matrix.user_ptrs[user_num]
        ep = self._matrix.user_ptrs[user_num + 1]
        inums = self._matrix.item_nums[sp:ep]
        fields = {}
        if self._matrix.ratings is not None:
            fields["rating"] = self._matrix.ratings[sp:ep]
        if self._matrix.timestamps is not None:
            fields["timestamp"] = self._matrix.timestamps[sp:ep]
        return ItemList(item_nums=inums, vocabulary=self.items, **fields)
