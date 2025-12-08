# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest import fixture, skip

from lenskit.data.msweb import load_ms_web
from lenskit.splitting import TTSplit

MSWEB_TRAIN = Path("data/anonymous-msweb.data.gz")
MSWEB_TEST = Path("data/anonymous-msweb.test.gz")


@fixture(scope="module")
def msweb() -> TTSplit:
    if not MSWEB_TRAIN.exists():
        skip("msweb not downloaded")
    if not MSWEB_TEST.exists():
        skip("msweb not downloaded")
    train = load_ms_web(MSWEB_TRAIN)
    test = load_ms_web(MSWEB_TEST, "collection")
    return TTSplit(train, test, "ms-web")
