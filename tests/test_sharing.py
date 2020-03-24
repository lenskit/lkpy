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
