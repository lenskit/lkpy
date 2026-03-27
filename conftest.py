# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import os
import warnings

import numpy as np
import structlog
import torch
from numpy.random import Generator, default_rng

from hypothesis import settings
from pytest import fixture, register_assert_rewrite, skip

from lenskit.parallel import ensure_parallel_init
from lenskit.random import init_global_rng

register_assert_rewrite("lenskit.testing")

# bring common fixtures into scope
from lenskit.testing import ml_100k, ml_ds, ml_ds_unchecked, ml_ratings  # noqa: E402, F401

logging.getLogger("numba").setLevel(logging.INFO)

_log = structlog.stdlib.get_logger("lenskit.tests")
RNG_SEED = 42
if "LK_TEST_FREE_RNG" in os.environ:
    warnings.warn("using nondeterministic RNG initialization")
    RNG_SEED = None

structlog.configure(
    [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.MaybeTimeStamper(fmt="iso"),
        structlog.processors.KeyValueRenderer(key_order=["timestamp", "event"]),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
os.environ["LK_SKIP_LOG_SETUP"] = "1"


@fixture
def fresh_rng() -> Generator:
    "Fixture to provide a fresh (non-fixed) RNG for tests that need it."
    seed = np.random.SeedSequence()
    return np.random.default_rng(seed)


@fixture
def rng() -> Generator:
    if RNG_SEED is None:
        return default_rng()
    else:
        return default_rng(RNG_SEED)


@fixture(autouse=True)
def init_rng(request):
    if RNG_SEED is not None:
        init_global_rng(RNG_SEED)


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
        rdm = item.get_closest_marker("realdata")
        slm = item.get_closest_marker("slow")
        if slm is None and (evm is not None or rdm is not None):
            _log.debug("adding slow mark to %s", item)
            item.add_marker("slow")


settings.register_profile("default", deadline=1000)
ensure_parallel_init()
