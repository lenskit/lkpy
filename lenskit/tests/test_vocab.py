"""
Tests for the Vocabulary class.
"""

import hypothesis.strategies as st
from hypothesis import given

from lenskit.data import Vocabulary


@given(st.lists(st.integers(min_value=1), unique=True))
def test_create_sized(keys):
    vocab = Vocabulary(keys)
    assert vocab.size == len(keys)
    assert len(vocab) == len(keys)


@given(st.lists(st.uuids(), unique=True))
def test_index(keys):
    vocab = Vocabulary(keys)
    assert vocab.size == len(keys)
    assert len(vocab) == len(keys)
