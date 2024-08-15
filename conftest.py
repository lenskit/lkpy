# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import os
import warnings

import torch
from seedbank import initialize, numpy_rng

from hypothesis import settings
from pytest import fixture, skip

from lenskit.parallel import ensure_parallel_init
from lenskit.util.test import ml_100k, ml_ds, ml_ratings  # noqa: F401

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


@fixture(scope="module", params=["cpu", "cuda"])
def torch_device(request):
    """
    Fixture for testing across Torch devices.  This fixture is parameterized, so
    if you write a test function with a parameter ``torch_device`` as its first
    parameter, it will be called once for each available Torch device.
    """
    dev = request.param
    if dev == "cuda" and not torch.cuda.is_available():
        skip("CUDA not available")
    if dev == "mps" and not torch.backends.mps.is_available():
        skip("MPS not available")
    yield dev


@fixture(autouse=True)
def log_test(request):
    try:
        modname = request.module.__name__ if request.module else "<unknown>"
    except Exception:
        modname = "<unknown>"
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


settings.register_profile("default", deadline=1000)
ensure_parallel_init()
