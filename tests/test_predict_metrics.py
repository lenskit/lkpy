import numpy as np
import pandas as pd
import os.path

from pytest import approx, raises, mark, skip

import lenskit.metrics.predict as pm


def test_check_missing_empty():
    pm._check_missing(pd.Series([]), 'error')
    # should pass
    assert True


def test_check_missing_has_values():
    pm._check_missing(pd.Series([1, 3, 2]), 'error')
    # should pass
    assert True


def test_check_missing_nan_raises():
    with raises(ValueError):
        pm._check_missing(pd.Series([1, np.nan, 3]), 'error')


def test_check_missing_raises():
    data = pd.Series([1, 7, 3], ['a', 'b', 'd'])
    ref = pd.Series([3, 2, 4], ['b', 'c', 'd'])
    ref, data = ref.align(data, join='left')
    with raises(ValueError):
        pm._check_missing(data, 'error')


def test_check_joined_ok():
    data = pd.Series([1, 7, 3], ['a', 'b', 'd'])
    ref = pd.Series([3, 2, 4], ['b', 'c', 'd'])
    ref, data = ref.align(data, join='inner')
    pm._check_missing(ref, 'error')
    # should get here
    assert True


def test_check_missing_ignore():
    data = pd.Series([1, 7, 3], ['a', 'b', 'd'])
    ref = pd.Series([3, 2, 4], ['b', 'c', 'd'])
    ref, data = ref.align(data, join='left')
    pm._check_missing(data, 'ignore')
    # should get here
    assert True


def test_rmse_one():
    rmse = pm.rmse([1], [1])
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = pm.rmse([1], [2])
    assert rmse == approx(1)

    rmse = pm.rmse([1], [0.5])
    assert rmse == approx(0.5)


def test_rmse_two():
    rmse = pm.rmse([1, 2], [1, 2])
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = pm.rmse([1, 1], [2, 2])
    assert rmse == approx(1)

    rmse = pm.rmse([1, 3], [3, 1])
    assert rmse == approx(2)

    rmse = pm.rmse([1, 3], [3, 2])
    assert rmse == approx(np.sqrt(5 / 2))


def test_rmse_array_two():
    rmse = pm.rmse(np.array([1, 2]), np.array([1, 2]))
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = pm.rmse(np.array([1, 1]), np.array([2, 2]))
    assert rmse == approx(1)

    rmse = pm.rmse(np.array([1, 3]), np.array([3, 1]))
    assert rmse == approx(2)


def test_rmse_series_two():
    rmse = pm.rmse(pd.Series([1, 2]), pd.Series([1, 2]))
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = pm.rmse(pd.Series([1, 1]), pd.Series([2, 2]))
    assert rmse == approx(1)

    rmse = pm.rmse(pd.Series([1, 3]), pd.Series([3, 1]))
    assert rmse == approx(2)


def test_rmse_series_subset_axis():
    rmse = pm.rmse(pd.Series([1, 3], ['a', 'c']), pd.Series([3, 4, 1], ['a', 'b', 'c']))
    assert rmse == approx(2)


def test_rmse_series_missing_value_error():
    with raises(ValueError):
        pm.rmse(pd.Series([1, 3], ['a', 'd']), pd.Series([3, 4, 1], ['a', 'b', 'c']))


def test_rmse_series_missing_value_ignore():
    rmse = pm.rmse(pd.Series([1, 3], ['a', 'd']), pd.Series([3, 4, 1], ['a', 'b', 'c']),
                   missing='ignore')
    assert rmse == approx(2)


def test_mae_two():
    mae = pm.mae([1, 2], [1, 2])
    assert isinstance(mae, float)
    assert mae == approx(0)

    mae = pm.mae([1, 1], [2, 2])
    assert mae == approx(1)

    mae = pm.mae([1, 3], [3, 1])
    assert mae == approx(2)

    mae = pm.mae([1, 3], [3, 2])
    assert mae == approx(1.5)


def test_mae_array_two():
    mae = pm.mae(np.array([1, 2]), np.array([1, 2]))
    assert isinstance(mae, float)
    assert mae == approx(0)

    mae = pm.mae(np.array([1, 1]), np.array([2, 2]))
    assert mae == approx(1)

    mae = pm.mae(np.array([1, 3]), np.array([3, 1]))
    assert mae == approx(2)


def test_mae_series_two():
    mae = pm.mae(pd.Series([1, 2]), pd.Series([1, 2]))
    assert isinstance(mae, float)
    assert mae == approx(0)

    mae = pm.mae(pd.Series([1, 1]), pd.Series([2, 2]))
    assert mae == approx(1)

    mae = pm.mae(pd.Series([1, 3]), pd.Series([3, 1]))
    assert mae == approx(2)


@mark.slow
def test_batch_rmse():
    import lk_test_utils as lktu
    import lenskit.crossfold as xf
    import lenskit.batch as batch
    import lenskit.algorithms.basic as bl

    if not os.path.exists('ml-100k/u.data'):
        raise skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    algo = bl.Bias(damping=5)

    def eval(train, test):
        model = algo.train(train)
        preds = batch.predict_pairs(lambda u, xs: algo.predict(model, u, xs), test)
        return preds.set_index(['user', 'item'])

    results = pd.concat((eval(train, test)
                         for (train, test)
                         in xf.partition_users(ratings, 5, xf.SampleN(5))))

    user_rmse = results.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))

    # we should have all users
    users = ratings.user.unique()
    assert len(user_rmse) == len(users)
    missing = np.setdiff1d(users, user_rmse.index)
    assert len(missing) == 0

    # we should not have any missing values
    assert all(user_rmse.notna())

    # we should have a reasonable mean
    assert user_rmse.mean() == approx(0.93, abs=0.05)
