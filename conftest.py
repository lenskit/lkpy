# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import os
import warnings

from seedbank import initialize, numpy_rng

from pytest import fixture

logging.getLogger("numba").setLevel(logging.INFO)

_log = logging.getLogger("lenskit.tests")
RNG_SEED = 42
if "LK_TEST_FREE_RNG" in os.environ:
    warnings.warn("using nondeterministic RNG initialization")
    RNG_SEED = None


@fixture
def rng():
    if RNG_SEED is None:
        return numpy_rng(os.urandom(4))
    else:
        return numpy_rng(RNG_SEED)


@fixture(autouse=True)
def init_rng(request):
    if RNG_SEED is None:
        initialize(os.urandom(4))
    else:
        initialize(RNG_SEED)


@fixture(autouse=True)
def log_test(request):
    modname = request.module.__name__ if request.module else "<unknown>"
    funcname = request.function.__name__ if request.function else "<unknown>"
    _log.info("running test %s:%s", modname, funcname)


def pytest_collection_modifyitems(items):
    # add 'slow' to all 'eval' tests
    for item in items:
        evm = item.get_closest_marker("eval")
        slm = item.get_closest_marker("slow")
        if evm is not None and slm is None:
            _log.debug("adding slow mark to %s", item)
            item.add_marker("slow")
