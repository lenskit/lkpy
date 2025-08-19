# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for the Vocabulary class.
"""

import pickle
from uuid import UUID, uuid4

import numpy as np
import pyarrow as pa

import hypothesis.strategies as st
from hypothesis import assume, given
from pytest import raises, warns

from lenskit.data import Vocabulary
from lenskit.diagnostics import DataWarning


def id_ints():
    li = np.iinfo(np.int64)
    return st.integers(li.min, li.max)


@given(
    st.one_of(
        st.sets(id_ints()),
        st.sets(st.emails()),
    )
)
def test_create_basic(keys: set[int | str]):
    vocab = Vocabulary(keys, reorder=True)
    assert vocab.size == len(keys)
    assert len(vocab) == len(keys)

    index = vocab.index
    assert all(index.values == sorted(keys))


@given(
    st.one_of(
        st.lists(id_ints()),
        st.lists(st.emails()),
    )
)
def test_create_nonunique(keys: list[int | str]):
    uq = set(keys)
    vocab = Vocabulary(keys, reorder=True)
    assert vocab.size == len(uq)
    assert len(vocab) == len(uq)

    index = vocab.index
    assert all(index.values == sorted(uq))


@given(
    st.one_of(
        st.lists(id_ints()),
        st.lists(st.emails()),
    )
)
def test_equal(keys: list[int | str]):
    vocab = Vocabulary(keys, reorder=True)

    v2 = Vocabulary(keys, reorder=True)
    assert v2 == vocab


@given(st.lists(id_ints()), st.sets(id_ints()))
def test_not_equal(keys: list[int], oks: set[int]):
    uq = set(keys)
    assume(oks != uq)

    vocab = Vocabulary(keys, reorder=True)
    v2 = Vocabulary(oks, reorder=True)
    try:
        assert v2 != vocab
    except AssertionError as e:
        e.add_note(f"v1 hash: {vocab._hash}")
        e.add_note(f"v2 hash: {v2._hash}")
        raise e


@given(
    st.one_of(
        st.sets(id_ints()),
        st.sets(st.emails()),
    ),
    st.lists(
        st.one_of(
            id_ints(),
            st.emails(),
        )
    ),
)
def test_contains(keys: set[int] | set[str], qs: set[int | str]):
    vocab = Vocabulary(keys, reorder=True)

    for qk in qs:
        if qk in keys:
            assert qk in vocab
        else:
            assert qk not in vocab


@given(
    st.one_of(
        st.sets(id_ints()),
        st.sets(st.emails()),
    )
)
def test_lookup_id_index(keys: set[int | str]):
    klist = sorted(keys)

    vocab = Vocabulary(keys, reorder=True)
    assert vocab.size == len(klist)
    assert len(vocab) == len(klist)

    # make sure the numbers are right
    assert all([vocab.number(k) == i for (i, k) in enumerate(klist)])

    # make sure the IDs are right
    assert all([vocab.term(i) == k for (i, k) in enumerate(klist)])


@given(
    st.one_of(
        st.sets(id_ints()),
        st.sets(st.emails()),
    ),
    st.one_of(id_ints(), st.emails()),
)
def test_lookup_bad_id(keys: set[int | str], key: int | str):
    assume(key not in keys)

    vocab = Vocabulary(keys, reorder=True)

    assert vocab.number(key, missing=None) is None

    with raises(KeyError):
        assert vocab.number(key, missing="error")


@given(
    st.one_of(
        st.sets(id_ints()),
        st.sets(st.emails()),
    ),
    st.one_of(id_ints()),
)
def test_lookup_bad_number(keys: set[int | str], num: int):
    assume(num < 0 or num >= len(keys))

    vocab = Vocabulary(keys, reorder=True)

    with raises(IndexError):
        vocab.term(num)


@given(
    st.sets(id_ints()),
    st.lists(id_ints()),
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
    st.sets(id_ints()),
    st.lists(id_ints()),
)
def test_lookup_many_nums_null(terms: set[int], lookup: list[int]):
    klist = sorted(terms)
    kpos = dict(zip(klist, range(len(klist))))

    vocab = Vocabulary(terms, reorder=True)

    nums = vocab.numbers(lookup, format="arrow", missing="null")
    assert len(nums) == len(lookup)
    assert nums.null_count == len([i for i in lookup if i not in terms])
    for n, k in zip(nums, lookup):
        if n.is_valid:
            assert n.as_py() == kpos[k]
        else:
            assert k not in terms


@given(
    st.sets(st.emails()),
    st.lists(st.emails()),
)
def test_lookup_many_nums_null(terms: set[str], lookup: list[str]):
    klist = sorted(terms)
    kpos = dict(zip(klist, range(len(klist))))

    vocab = Vocabulary(terms, reorder=True)

    nums = vocab.numbers(lookup, format="arrow", missing="null")
    assert len(nums) == len(lookup)
    assert nums.null_count == len([i for i in lookup if i not in terms])
    for n, k in zip(nums, lookup):
        if n.is_valid:
            assert n.as_py() == kpos[k]
        else:
            assert k not in terms


@given(
    st.data(),
    st.sets(id_ints()),
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


@given(st.one_of(st.sets(id_ints()), st.sets(st.emails())))
def test_all_terms(initial: set[int] | set[str]):
    vocab = Vocabulary(initial)

    tl = sorted(initial)

    terms = vocab.terms()
    assert isinstance(terms, np.ndarray)
    assert all(terms == tl)


def test_lots_of_strings(rng: np.random.Generator):
    "make sure a lot of strings work"
    N = 10_000

    ids = [str(uuid4()) for _i in range(N)]
    vocab = Vocabulary(ids, reorder=True)

    arr = vocab.id_array()
    assert len(arr) == N
    assert arr.null_count == 0

    id_arr = arr.to_numpy(zero_copy_only=False)

    q_pos = rng.choice(N, 5000, replace=True)
    query = id_arr[q_pos]

    nums = vocab.numbers(query, format="arrow", missing="null")
    assert nums.type == pa.int32()
    assert nums.null_count == 0
    assert np.all(nums.to_numpy() >= 0)
    assert np.all(nums.to_numpy() < N)
    assert np.all(id_arr[nums.to_numpy()] == query)


@given(st.one_of(st.lists(id_ints()), st.lists(st.emails()), st.lists(st.emails())))
def test_pickle(initial: list[int | str]):
    vocab = Vocabulary(initial, reorder=True)

    blob = pickle.dumps(vocab)
    v2 = pickle.loads(blob)

    assert v2 is not vocab
    assert len(v2) == len(vocab)
    assert np.all(v2.ids() == vocab.ids())
    assert v2 == vocab
