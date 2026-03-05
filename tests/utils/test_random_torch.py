# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import SeedSequence

from hypothesis import given
from hypothesis import strategies as st
from pytest import importorskip

from lenskit.random import random_generator

torch = importorskip("torch")

if TYPE_CHECKING:
    import torch


def test_torch_rng_none():
    rng = random_generator(type="torch")
    assert isinstance(rng, torch.Generator)


@given(st.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max))
def test_torch_rng_int(seed):
    rng = random_generator(seed, type="torch")
    assert isinstance(rng, torch.Generator)


@given(st.lists(st.integers(0)))
def test_torch_rng_seq(seed):
    rng = random_generator(seed, type="torch")
    assert isinstance(rng, torch.Generator)


@given(st.integers(0))
def test_torch_rng_seed(seed):
    seed = SeedSequence(seed)
    rng = random_generator(seed, type="torch")
    assert isinstance(rng, torch.Generator)


def test_torch_rng_from_numpy(rng):
    trng = random_generator(rng, type="torch")
    assert isinstance(trng, torch.Generator)


def test_torch_rng_from_numpy_bits(rng):
    trng = random_generator(rng.bit_generator, type="torch")
    assert isinstance(trng, torch.Generator)
