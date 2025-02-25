# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
"""
Multi-format array layout.
"""

# pyright: basic
from __future__ import annotations

import numpy as np
import pyarrow as pa
import torch
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Generic, Literal, LiteralString, Sequence, TypeVar, overload

NPT = TypeVar("NPT", bound=np.generic)


class MTArray(Generic[NPT]):
    """
    Multi-typed array class, allowing arrays to be easily converted between
    NumPy, PyTorch, and other supported backends, caching the conversion result.

    We use this class instead of one canonical format so that data can be
    converted lazily, and can be left in-place when passing from one component
    to another that use the same computational engine.

    .. note::

        This class is only intended for read-only arrays.  It is **not defined**
        whether the different arrays share storage, and modifying one format may
        or may not modify another.  For example, PyTorch and NumPy usually share
        storage when both are on CPU, but a GPU tensor and CPU ndarray do not.

    Stability:
        Internal
    """

    _shape: tuple[int, ...] | None = None
    _unknown: object = None
    _numpy: NDArray[NPT] | None = None
    _arrow: pa.Array | pa.Tensor | None = None
    _torch: torch.Tensor | None = None

    def __init__(
        self, array: NDArray[NPT] | torch.Tensor | pa.Array | pa.Tensor | Sequence | ArrayLike
    ):
        """
        Construct a new MTArray around an array.
        """
        # TODO: support DLpack
        if isinstance(array, torch.Tensor):
            # torch might not be on-device
            self._torch = array
            self._shape = array.shape
        else:
            # stash it in the common-format field for lazy conversion
            self._unknown = array
            # save the shape for well-known types
            if isinstance(array, pa.Array):
                self._shape = (len(array),)
            elif isinstance(array, (pa.Tensor, np.ndarray)):
                self._shape = array.shape

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            self._shape = self.numpy().shape

        return self._shape

    def numpy(self) -> NDArray[NPT]:
        """
        Get the array as a NumPy array.
        """
        if self._numpy is None:
            self._numpy = np.asarray(self._convertible())

        assert self._numpy is not None
        return self._numpy

    def torch(self, *, device: str | None = None) -> torch.Tensor:
        """
        Get the array as a PyTorch tensor.

        Args:
            device:
                The device on which the Torch tensor should reside.
        """
        if self._torch is None:
            if torch.is_tensor(self._unknown):
                self._torch = self._unknown
            else:
                # Make sure we have a writeable array for Torch. Client code
                # still shouldn't write to it.
                arr = np.require(self.numpy(), requirements="W")
                return torch.tensor(arr)

        if device:
            return self._torch.to(device)
        else:
            return self._torch

    def arrow(self) -> pa.Array | pa.Tensor:
        """
        Get the array as an Arrow :class:`~pyarrow.Array`.
        """
        if self._arrow is None:
            if isinstance(self._unknown, (pa.Array, pa.Tensor)):
                self._arrow = self._unknown
            else:
                arr = self.numpy()
                if len(arr.shape) == 1:  # type: ignore
                    self._arrow = pa.array(arr)  # type: ignore
                else:
                    self._arrow = pa.Tensor.from_numpy(arr)  # type: ignore

        assert self._arrow is not None
        return self._arrow

    @overload
    def to(self, format: Literal["numpy"]) -> NDArray[NPT]: ...
    @overload
    def to(self, format: Literal["torch"], *, device: str | None) -> torch.Tensor: ...
    @overload
    def to(self, format: Literal["arrow"]) -> pa.Array | pa.Tensor: ...
    @overload
    def to(self, format: LiteralString, *, device: str | None = None) -> ArrayLike: ...
    def to(self, format: str, *, device: str | None = None) -> object:
        """
        Obtain the array in the specified format (dynamic version).
        """
        match format:
            case "numpy":
                return self.numpy()
            case "torch":
                return self.torch(device=device)
            case "arrow":
                return self.arrow()
            case _:
                raise RuntimeError(f"unsupported array format {format}")

    def _convertible(self) -> object:
        """
        Get the data suitable for passing to an ``as_array`` method.
        """
        # look for a good format for the data. if we've already made a numpy,
        # use that; otherwise, try unknown, and fall back to torch (moved to
        # CPU). end result: convertible data.
        if self._numpy is not None:
            return self._numpy
        elif self._unknown is not None:
            return self._unknown
        elif self._torch is not None:
            return self._torch.cpu()
        else:
            raise RuntimeError("cannot find array data")

    def __array__(self, dtype=None, copy=None) -> NDArray[NPT]:
        arr = self.numpy()
        if dtype is not None:
            arr = np.require(arr, dtype)
        if copy:
            return arr.copy()
        else:
            return arr


MTFloatArray = MTArray[np.floating]
MTIntArray = MTArray[np.integer]
MTGenericArray = MTArray[np.generic]
