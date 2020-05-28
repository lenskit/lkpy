import os

import pickle
import numpy as np

import lenskit.util.test as lktu
from lenskit import sharing as lks
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import BiasedMF
from lenskit.algorithms.item_knn import ItemItem

from pytest import mark

stores = [lks.FileModelStore]
if pickle.HIGHEST_PROTOCOL >= 5:
    # we have Python 3.8
    stores.append(lks.SHMModelStore)

store_param = mark.parametrize('store_cls', stores)


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


@mark.skipif(lks.shm is None, reason='shared_memory not available')
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
    os.environ['LK_TEMP_DIR'] = os.fspath(tmp_path)
    try:
        share = lks.persist(matrix)
        assert isinstance(share, lks.BPKPersisted)
    finally:
        del os.environ['LK_TEMP_DIR']

    try:
        m2 = share.get()
        assert m2 is not matrix
        assert np.all(m2 == matrix)
        del m2
    finally:
        share.close()


@store_param
def test_store_init(store_cls):
    "Test that a store initializes and shuts down."
    with store_cls():
        pass


@store_param
def test_store_save(store_cls):
    algo = Popular()
    algo.fit(lktu.ml_test.ratings)

    with store_cls() as store:
        k = store.put_model(algo)
        with store.get_model(k) as a2:
            assert a2 is not algo
            assert a2.item_pop_ is not algo.item_pop_
            assert all(a2.item_pop_ == algo.item_pop_)
            del a2


@store_param
def test_store_client(store_cls):
    algo = Popular()
    algo.fit(lktu.ml_test.ratings)

    with store_cls() as store:
        k = store.put_model(algo)
        client = store.client()

        with client.get_model(k) as a2:
            assert a2 is not algo
            assert a2.item_pop_ is not algo.item_pop_
            assert all(a2.item_pop_ == algo.item_pop_)
            del a2


@store_param
def test_store_client_pickle(store_cls):
    algo = Popular()
    algo.fit(lktu.ml_test.ratings)

    with store_cls() as store:
        k = store.put_model(algo)
        client = store.client()
        client = pickle.loads(pickle.dumps(client))
        k = pickle.loads(pickle.dumps(k))

        with client.get_model(k) as a2:
            assert a2 is not algo
            assert a2.item_pop_ is not algo.item_pop_
            assert all(a2.item_pop_ == algo.item_pop_)
            del a2


@lktu.wantjit
@store_param
def test_store_als(store_cls):
    algo = BiasedMF(10)
    algo.fit(lktu.ml_test.ratings)

    with store_cls() as store:
        k = store.put_model(algo)
        client = store.client()

        with client.get_model(k) as a2:
            assert a2 is not algo
            assert a2.item_features_ is not algo.item_features_
            assert np.all(a2.item_features_ == algo.item_features_)
            assert a2.user_features_ is not algo.user_features_
            assert np.all(a2.user_features_ == algo.user_features_)
            del a2


@lktu.wantjit
@store_param
def test_store_iknn(store_cls):
    algo = ItemItem(10)
    algo = Recommender.adapt(algo)
    algo.fit(lktu.ml_test.ratings)

    with store_cls() as store:
        k = store.put_model(algo)
        client = store.client()

        with client.get_model(k) as a2:
            assert a2 is not algo
            del a2


def test_create_store():
    with lks.get_store() as store:
        assert len(lks._active_stores()) == 1
        assert lks._active_stores()[-1] is store


def test_reuse_store():
    with lks.get_store() as outer:
        with lks.get_store() as inner:
            assert inner is outer
            assert len(lks._active_stores()) == 2
            assert lks._active_stores()[-1] is inner
        assert len(lks._active_stores()) == 1


def test_new_store():
    with lks.get_store() as outer:
        with lks.get_store(False) as inner:
            assert inner is not outer
            assert len(lks._active_stores()) == 2
            assert lks._active_stores()[-1] is inner
            assert lks._active_stores()[0] is outer
        assert len(lks._active_stores()) == 1
