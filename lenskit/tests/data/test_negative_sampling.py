import numpy as np

from lenskit.data import Dataset


def test_negative(rng: np.random.Generator, ml_ds: Dataset):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 100, replace=True)
    users = np.require(users, "i4")

    negatives = matrix.negative_items(users, rng=rng)

    assert np.all(negatives >= 0)
    assert np.all(negatives < ml_ds.item_count)
    for u, i in zip(users, negatives):
        row = ml_ds.user_row(user_num=u)
        assert i not in row.numbers()


def test_negative_unverified(rng: np.random.Generator, ml_ds: Dataset):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 500, replace=True)
    users = np.require(users, "i4")

    negatives = matrix.negative_items(users, verify=False, rng=rng)

    assert np.all(negatives >= 0)
    assert np.all(negatives < ml_ds.item_count)
