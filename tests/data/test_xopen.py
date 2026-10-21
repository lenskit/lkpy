# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import gzip
import sys
from pathlib import Path

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

from lenskit.data import xopen


def test_xopen_plain(tmp_path: Path):
    with xopen(tmp_path / "test.txt", "wt") as f:
        f.write("hello")
    assert (tmp_path / "test.txt").read_text("utf8") == "hello"


def test_xopen_gzip(tmp_path: Path):
    with xopen(tmp_path / "test.txt.gz", "wt") as f:
        f.write("hello")
    data = (tmp_path / "test.txt.gz").read_bytes()
    assert gzip.decompress(data) == b"hello"


def test_xopen_zstd(tmp_path: Path):
    with xopen(tmp_path / "test.txt.zst", "wt") as f:
        f.write("hello")
    data = (tmp_path / "test.txt.zst").read_bytes()
    assert zstd.decompress(data) == b"hello"
