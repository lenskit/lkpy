import numpy as np
import torch

from pytest import raises

from lenskit.data import ItemList
from lenskit.data.vocab import Vocabulary

ITEMS = ["a", "b", "c", "d", "e"]
VOCAB = Vocabulary(ITEMS)


def test_empty():
    il = ItemList()

    assert len(il) == 0
    assert il.numbers().shape == (0,)
    assert il.ids().shape == (0,)
    assert il.scores() is None


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
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB)

    assert len(il) == 5
    assert il.numbers().shape == (5,)
    assert il.ids().shape == (5,)

    assert all(il.numbers() == np.arange(5))
    assert all(il.ids() == ITEMS)


def test_item_id_list_vocab():
    il = ItemList(item_ids=ITEMS, vocabulary=VOCAB)

    assert len(il) == 5
    assert il.numbers().shape == (5,)
    assert il.ids().shape == (5,)

    assert all(il.numbers() == np.arange(5))
    assert all(il.ids() == ITEMS)


def test_scores():
    data = np.random.randn(5)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)

    scores = il.scores()
    assert scores is not None
    assert scores.shape == (5,)
    assert np.all(scores == data)

    st = il.scores("torch")
    assert isinstance(st, torch.Tensor)
    assert st.shape == (5,)
    assert np.all(st.numpy() == data)
