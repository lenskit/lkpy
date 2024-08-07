# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for the Vocabulary class.
"""

from uuid import UUID

import numpy as np

import hypothesis.strategies as st
from hypothesis import assume, given
from pytest import raises

from lenskit.data import Vocabulary


@given(
    st.one_of(
        st.sets(st.integers()),
        st.sets(st.uuids()),
    )
)
def test_create_basic(keys: set[int | str | UUID]):
    vocab = Vocabulary(keys)
    assert vocab.size == len(keys)
    assert len(vocab) == len(keys)

    index = vocab.index
    assert all(index.values == sorted(keys))


@given(
    st.one_of(
        st.lists(st.integers()),
        st.lists(st.uuids()),
    )
)
def test_create_nonunique(keys: list[int | str | UUID]):
    uq = set(keys)
    vocab = Vocabulary(keys)
    assert vocab.size == len(uq)
    assert len(vocab) == len(uq)

    index = vocab.index
    assert all(index.values == sorted(uq))


@given(
    st.one_of(
        st.lists(st.integers()),
        st.lists(st.uuids()),
    )
)
def test_equal(keys: list[int | str | UUID]):
    vocab = Vocabulary(keys)

    v2 = Vocabulary(keys)
    assert v2 == vocab


@given(st.lists(st.integers()), st.sets(st.integers()))
def test_not_equal(keys: list[int], oks: set[int]):
    uq = set(keys)
    assume(oks != uq)

    vocab = Vocabulary(keys)
    v2 = Vocabulary(oks)
    assert v2 != vocab


@given(
    st.one_of(
        st.sets(st.integers()),
        st.sets(st.uuids()),
    ),
    st.lists(
        st.one_of(
            st.integers(),
            st.uuids(),
        )
    ),
)
def test_contains(keys: set[int] | set[str] | set[UUID], qs: set[int | str | UUID]):
    vocab = Vocabulary(keys)

    for qk in qs:
        if qk in keys:
            assert qk in vocab
        else:
            assert qk not in vocab


@given(
    st.one_of(
        st.sets(st.integers()),
        st.sets(st.uuids()),
    )
)
def test_lookup_id_index(keys: set[int | str | UUID]):
    klist = sorted(keys)

    vocab = Vocabulary(keys)
    assert vocab.size == len(klist)
    assert len(vocab) == len(klist)

    # make sure the numbers are right
    assert all([vocab.number(k) == i for (i, k) in enumerate(klist)])

    # make sure the IDs are right
    assert all([vocab.term(i) == k for (i, k) in enumerate(klist)])


@given(
    st.one_of(
        st.sets(st.integers()),
        st.sets(st.uuids()),
    ),
    st.one_of(st.integers(), st.uuids()),
)
def test_lookup_bad_id(keys: set[int | str | UUID], key: int | str | UUID):
    assume(key not in keys)

    vocab = Vocabulary(keys)

    assert vocab.number(key, missing=None) is None

    with raises(KeyError):
        assert vocab.number(key, missing="error")


@given(
    st.one_of(
        st.sets(st.integers()),
        st.sets(st.uuids()),
    ),
    st.one_of(st.integers()),
)
def test_lookup_bad_number(keys: set[int | str | UUID], num: int):
    assume(num < 0 or num >= len(keys))

    vocab = Vocabulary(keys)

    with raises(IndexError):
        vocab.term(num)


@given(
    st.sets(st.integers()),
    st.lists(st.integers()),
)
def test_lookup_many_nums(terms: set[int], lookup: list[int]):
    klist = sorted(terms)
    kpos = dict(zip(klist, range(len(klist))))

    vocab = Vocabulary(terms)

    nums = vocab.numbers(lookup, missing="negative")
    assert len(nums) == len(lookup)
    for n, k in zip(nums, lookup):
        if n < 0:
            assert k not in terms
        else:
            assert n == kpos[k]


@given(
    st.data(),
    st.sets(st.integers()),
)
def test_lookup_many_terms(data, terms: set[int]):
    assume(len(terms) > 0)
    lookup = data.draw(st.lists(st.integers(min_value=0, max_value=len(terms) - 1)))
    klist = sorted(terms)

    vocab = Vocabulary(terms)

    keys = vocab.terms(lookup)
    assert len(keys) == len(lookup)
    for k, n in zip(keys, lookup):
        assert k == klist[n]


@given(st.sets(st.integers()), st.lists(st.integers()))
def test_add_terms(initial: set[int], new: list[int]):
    ni = len(initial)
    vocab = Vocabulary(initial)

    fresh = set(t for t in new if t not in initial)
    flist = sorted(fresh)

    vocab.add_terms(new)
    assert vocab.size == ni + len(fresh)

    assert all(vocab.number(k) == i for (i, k) in enumerate(sorted(initial)))
    assert all(vocab.number(k) == i + ni for (i, k) in enumerate(flist))


@given(st.one_of(st.sets(st.integers()), st.sets(st.uuids())))
def test_all_terms(initial: set[int] | set[str]):
    vocab = Vocabulary(initial)

    tl = sorted(initial)

    terms = vocab.terms()
    assert isinstance(terms, np.ndarray)
    assert all(terms == tl)
