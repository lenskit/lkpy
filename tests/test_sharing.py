import pickle

import lenskit.util.test as lktu
from lenskit import sharing as lks
from lenskit.algorithms.basic import Popular

from pytest import mark

stores = [lks.JoblibModelStore]
store_param = mark.parametrize('store_cls', stores)


def test_sharing_mode():
    "Ensure sharing mode decorator turns on sharing"
    assert not lks.in_share_context()

    with lks.sharing_mode():
        assert lks.in_share_context()

    assert not lks.in_share_context()


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
        a2 = store.get_model(k)
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

        a2 = client.get_model(k)
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

        a2 = client.get_model(k)
        assert a2 is not algo
        assert a2.item_pop_ is not algo.item_pop_
        assert all(a2.item_pop_ == algo.item_pop_)
        del a2


def test_create_store():
    with lks.get_store() as store:
        assert len(lks._active_stores) == 1
        assert lks._active_stores[-1] is store


def test_reuse_store():
    with lks.get_store() as outer:
        with lks.get_store() as inner:
            assert inner is outer
            assert len(lks._active_stores) == 2
            assert lks._active_stores[-1] is inner
        assert len(lks._active_stores) == 1


def test_new_store():
    with lks.get_store() as outer:
        with lks.get_store(False) as inner:
            assert inner is not outer
            assert len(lks._active_stores) == 2
            assert lks._active_stores[-1] is inner
            assert lks._active_stores[0] is outer
        assert len(lks._active_stores) == 1
