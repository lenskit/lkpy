# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os

import pickle
import numpy as np

import lenskit.util.test as lktu
from lenskit import sharing as lks
from lenskit.algorithms.als import BiasedMF

from pytest import mark


def test_sharing_mode():
    "Ensure sharing mode decorator turns on sharing"
    assert not lks.in_share_context()

    with lks.sharing_mode():
        assert lks.in_share_context()

    assert not lks.in_share_context()


def test_persist_bpk():
    matrix = np.random.randn(1000, 100)
    share = lks.persist_binpickle(matrix)
    try:
        assert share.path.exists()
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


@mark.skipif(not lks.SHM_AVAILABLE, reason="shared_memory not available")
def test_persist_shm():
    matrix = np.random.randn(1000, 100)
    share = lks.persist_shm(matrix)
    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


def test_persist():
    "Test default persistence"
    matrix = np.random.randn(1000, 100)
    share = lks.persist(matrix)
    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


def test_persist_dir(tmp_path):
    "Test persistence with a configured directory"
    matrix = np.random.randn(1000, 100)
    with lktu.set_env_var("LK_TEMP_DIR", os.fspath(tmp_path)):
        share = lks.persist(matrix)
        assert isinstance(share, lks.BPKPersisted)

    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


def test_persist_method():
    "Test persistence with a specified method"
    matrix = np.random.randn(1000, 100)

    share = lks.persist(matrix, method="binpickle")
    assert isinstance(share, lks.BPKPersisted)

    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


def test_store_als():
    algo = BiasedMF(10)
    algo.fit(lktu.ml_test.ratings)

    shared = lks.persist(algo)
    k2 = pickle.loads(pickle.dumps(shared))
    try:
        a2 = k2.get()

        assert a2 is not algo
        assert a2.item_features_ is not algo.item_features_
        assert np.all(a2.item_features_ == algo.item_features_)
        assert a2.user_features_ is not algo.user_features_
        assert np.all(a2.user_features_ == algo.user_features_)
        del a2
        k2.close()
        del k2
    finally:
        shared.close()
