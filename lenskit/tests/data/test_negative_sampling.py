import numpy as np

from lenskit.data import Dataset
from lenskit.logging import get_logger

_log = get_logger(__name__)


def test_negative(rng: np.random.Generator, ml_ds: Dataset):
    log = _log.bind()
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 100, replace=True)
    users = np.require(users, "i4")

    negatives = matrix.sample_negatives(users, rng=rng)

    log.info("checking basic item results")
    assert np.all(negatives >= 0)
    assert np.all(negatives < ml_ds.item_count)
    log.info("checking negative items")
    for u, i in zip(users, negatives):
        ulog = log.bind(
            user_num=u.item(),
            user_id=int(ml_ds.users.id(u)),  # type: ignore
            item_num=i.item(),
            item_id=int(ml_ds.items.id(i)),  # type: ignore
        )
        row = ml_ds.user_row(user_num=u)
        ulog = ulog.bind(u_nitems=len(row))
        ulog.debug("checking if item is negative")
        assert (u, i) not in matrix.rc_index
        print(ml_ds.users.id(u), row.ids())
        assert i not in row.numbers()


def test_negative_unverified(rng: np.random.Generator, ml_ds: Dataset):
    matrix = ml_ds.interactions().matrix()

    users = rng.choice(ml_ds.user_count, 500, replace=True)
    users = np.require(users, "i4")

    negatives = matrix.sample_negatives(users, verify=False, rng=rng)

    assert np.all(negatives >= 0)
    assert np.all(negatives < ml_ds.item_count)
