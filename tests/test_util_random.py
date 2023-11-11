import zlib
import numpy as np
from lenskit.util import random
from seedbank import root_seed


def test_generator():
    rng = random.rng()
    assert isinstance(rng, np.random.Generator)


def test_generator_seed():
    rng = random.rng(42)
    assert isinstance(rng, np.random.Generator)


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
    rng = random.rng(rng1, legacy=True)
    assert isinstance(rng, np.random.RandomState)


def test_generator_legacy_ss():
    seq = np.random.SeedSequence(42)
    rng = random.rng(seq, legacy=True)
    assert isinstance(rng, np.random.RandomState)


def test_generator_convert_to_legacy():
    rng1 = random.rng()
    rng = random.rng(rng1, legacy=True)
    assert isinstance(rng, np.random.RandomState)


def test_generator_passthrough():
    rng1 = random.rng()
    rng = random.rng(rng1)
    assert isinstance(rng, np.random.Generator)
    assert rng is rng1


def test_initialize():
    random.init_rng(42)
    assert root_seed().entropy == 42
    assert len(root_seed().spawn_key) == 0


def test_initialize_key():
    random.init_rng(42, "wombat")
    assert root_seed().entropy == 42
    # assert root_seed().spawn_key == (zlib.crc32(b'wombat'),)


def test_derive_seed():
    random.init_rng(42, propagate=False)
    s2 = random.derive_seed()
    assert s2.entropy == 42
    assert s2.spawn_key == (0,)


def test_derive_seed_intkey():
    random.init_rng(42, propagate=False)
    s2 = random.derive_seed(10, 7)
    assert s2.entropy == 42
    assert s2.spawn_key == (10, 7)


def test_derive_seed_str():
    random.init_rng(42, propagate=False)
    s2 = random.derive_seed(b"wombat")
    assert s2.entropy == 42
    # assert s2.spawn_key == (zlib.crc32(b'wombat'),)
