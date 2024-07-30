import numpy as np

from pytest import raises

from lenskit.data import ItemList
from lenskit.data.vocab import Vocabulary


def test_empty():
    il = ItemList()

    assert len(il) == 0
    assert il.numbers().shape == (0,)
    assert il.ids().shape == (0,)


def test_item_list():
    il = ItemList(item_ids=["one", "two"])

    assert len(il) == 2
    assert il.ids().shape == (2,)

    with raises(RuntimeError, match="item numbers not available"):
        il.numbers()


def test_item_num_list():
    il = ItemList(item_nums=np.arange(5))

    assert len(il) == 5
    assert il.numbers().shape == (5,)

    with raises(RuntimeError, match="item IDs not available"):
        il.ids()


def test_item_num_list_vocab():
    il = ItemList(item_nums=np.arange(5), vocabulary=Vocabulary(["a", "b", "c", "d", "e"]))

    assert len(il) == 5
    assert il.numbers().shape == (5,)
    assert il.ids().shape == (5,)

    assert all(il.numbers() == np.arange(5))
    assert all(il.ids() == ["a", "b", "c", "d", "e"])


def test_item_id_list_vocab():
    idl = ["a", "b", "c", "d", "e"]
    il = ItemList(item_ids=idl, vocabulary=Vocabulary(idl))

    assert len(il) == 5
    assert il.numbers().shape == (5,)
    assert il.ids().shape == (5,)

    assert all(il.numbers() == np.arange(5))
    assert all(il.ids() == ["a", "b", "c", "d", "e"])
