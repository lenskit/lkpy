# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
from __future__ import annotations

import numpy as np
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
    """

    _shape: tuple[int, ...] | None = None
    _unknown: object = None
    _numpy: NDArray[NPT] | None = None
    _torch: torch.Tensor | None = None

    def __init__(self, array: NDArray[NPT] | torch.Tensor | Sequence | ArrayLike):
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
            self._torch = torch.as_tensor(self._convertible())

        if device:
            return self._torch.to(device)
        else:
            return self._torch

    @overload
    def to(self, format: Literal["numpy"]) -> NDArray[NPT]: ...
    @overload
    def to(self, format: Literal["torch"], *, device: str | None) -> torch.Tensor: ...
    @overload
    def to(self, format: LiteralString, *, device: str | None = None) -> ArrayLike: ...
    def to(self, format: str, *, device: str | None = None) -> ArrayLike:
        """
        Obtain the array in the specified format (dynamic version).
        """
        if format == "numpy":
            return self.numpy()
        elif format == "torch":
            return self.torch(device=device)
        else:
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
