import numpy as np
from lenskit.util import random

from pytest import mark

new_gen = mark.skipif(not random._have_gen, reason="requires NumPy with generators")


@new_gen
def test_generator():
    rng = random.rng()
    assert isinstance(rng, np.random.Generator)


@new_gen
def test_generator_seed():
    rng = random.rng(42)
    assert isinstance(rng, np.random.Generator)


@new_gen
def test_generator_seed_seq():
    seq = np.random.SeedSequence(42)
    rng = random.rng(seq)
    assert isinstance(rng, np.random.Generator)


def test_generator_legacy():
    rng = random.rng(legacy=True)
    assert isinstance(rng, np.random.RandomState)


def test_generator_legacy_seed():
    rng = random.rng(42, legacy=True)
    assert isinstance(rng, np.random.RandomState)


def test_generator_legacy_passthrough():
    rng1 = random.rng(legacy=True)
    rng = random.rng(rng1)
    assert isinstance(rng, np.random.RandomState)


@new_gen
def test_generator_legacy_ss():
    seq = np.random.SeedSequence(42)
    rng = random.rng(seq, legacy=True)
    assert isinstance(rng, np.random.RandomState)


@new_gen
def test_generator_convert_to_legacy():
    rng1 = random.rng()
    rng = random.rng(rng1, legacy=True)
    assert isinstance(rng, np.random.RandomState)


@new_gen
def test_generator_passthrough():
    rng1 = random.rng()
    rng = random.rng(rng1)
    assert isinstance(rng, np.random.Generator)
    assert rng is rng1
