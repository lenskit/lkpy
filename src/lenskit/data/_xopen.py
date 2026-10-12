# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import gzip
import lzma
import sys
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Literal, TextIO, overload

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd


@overload
def xopen(
    path: Path | PathLike[str] | str, mode: Literal["r", "rb", "w", "wb", "a", "ab"] = "r"
) -> BinaryIO: ...
@overload
def xopen(path: Path | PathLike[str] | str, mode: Literal["rt", "wt", "at"]) -> TextIO: ...
def xopen(path: Path | PathLike[str] | str, mode: str = "rb"):
    """
    Open a file with the appropriate compression mode.

    This is modeled after the ``xopen`` package, but implemented directly
    without as many external dependencies, preferring Python standard library
    modules.

    .. note::

        Unlike :func:`open`, but like :func:`gzip.open` and friends, the default
        mode is binary.

    Args:
        path: The path to the file to open.
        mode: The mode to open the file in.
    """
    if mode in ("r", "w", "a"):
        mode += "b"

    path = Path(path)
    match path.suffix:
        case ".gz":
            return gzip.open(path, mode)
        case ".xz":
            return lzma.open(path, mode)
        case ".zst" | ".zstd":
            return zstd.open(path, mode)  # ty: ignore[no-matching-overload]
        case _:
            return open(path, mode)
