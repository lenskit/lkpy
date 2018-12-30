from lenskit.algorithms import basic

import pandas as pd
import numpy as np

import lk_test_utils as lktu
from pytest import approx, mark

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_memorized():
    algo = basic.Memorized(simple_df)

    preds = algo.predict_for_user(10, [1, 2])
    assert set(preds.index) == set([1, 2])
    assert all(preds == pd.Series({1: 4.0, 2: 5.0}))

    preds = algo.predict_for_user(12, [1, 3])
    assert set(preds.index) == set([1, 3])
    assert preds.loc[1] == 3.0
    assert np.isnan(preds.loc[3])


def test_memorized_batch():
    algo = basic.Memorized(simple_df)

    preds = algo.predict(pd.DataFrame({'user': [10, 10, 12], 'item': [1, 2, 1]}))
    assert isinstance(preds, pd.Series)
    assert preds.name == 'prediction'
    assert set(preds.index) == set([0, 1, 2])
    assert all(preds == [4.0, 5.0, 3.0])


def test_memorized_batch_ord():
    algo = basic.Memorized(simple_df)

    preds = algo.predict(pd.DataFrame({'user': [10, 12, 10], 'item': [1, 1, 2]}))
    assert set(preds.index) == set([0, 1, 2])
    assert all(preds == [4.0, 3.0, 5.0])


def test_memorized_batch_missing():
    algo = basic.Memorized(simple_df)

    preds = algo.predict(pd.DataFrame({'user': [10, 12, 12], 'item': [1, 1, 3]}))
    assert set(preds.index) == set([0, 1, 2])
    assert all(preds.iloc[:2] == [4.0, 3.0])
    assert np.isnan(preds.iloc[2])


def test_memorized_batch_keep_index():
    algo = basic.Memorized(simple_df)

    query = pd.DataFrame({'user': [10, 10, 12], 'item': [1, 2, 1]},
                         index=np.random.choice(np.arange(10), 3, False))
    preds = algo.predict(query)
    assert all(preds.index == query.index)
    assert all(preds == [4.0, 5.0, 3.0])

def test_fallback_train_one():
    algo = basic.Fallback(basic.Bias())
    algo.fit(lktu.ml_pandas.renamed.ratings)
    assert len(algo.algorithms) == 1
    assert isinstance(algo.algorithms[0], basic.Bias)
    assert algo.algorithms[0].mean_ == approx(lktu.ml_pandas.ratings.rating.mean())


def test_fallback_train_one_pred_impossible():
    algo = basic.Fallback(basic.Memorized(simple_df))
    algo.fit(lktu.ml_pandas.renamed.ratings)

    preds = algo.predict_for_user(10, [1, 2])
    assert set(preds.index) == set([1, 2])
    assert all(preds == pd.Series({1: 4.0, 2: 5.0}))

    preds = algo.predict_for_user(12, [1, 3])
    assert set(preds.index) == set([1, 3])
    assert preds.loc[1] == 3.0
    assert np.isnan(preds.loc[3])


def test_fallback_predict():
    algo = basic.Fallback(basic.Memorized(simple_df), basic.Bias())
    algo.fit(lktu.ml_pandas.renamed.ratings)
    assert len(algo.algorithms) == 2

    bias = algo.algorithms[1]
    assert isinstance(bias, basic.Bias)
    assert bias.mean_ == approx(lktu.ml_pandas.ratings.rating.mean())

    # first user + item
    preds = algo.predict_for_user(10, [1])
    assert preds.loc[1] == 4.0
    # second user + first item
    preds = algo.predict_for_user(15, [1])
    assert preds.loc[1] == approx(bias.mean_ + bias.user_offsets_.loc[15] + bias.item_offsets_.loc[1])

    # second item + user item
    preds = algo.predict_for_user(12, [2])
    assert preds.loc[2] == approx(bias.mean_ + bias.user_offsets_.loc[12] + bias.item_offsets_.loc[2])

    # blended
    preds = algo.predict_for_user(10, [1, 5])
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(bias.mean_ + bias.user_offsets_.loc[10] + bias.item_offsets_.loc[5])

    # blended unknown
    preds = algo.predict_for_user(10, [5, 1, -23081])
    assert len(preds) == 3
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(bias.mean_ + bias.user_offsets_.loc[10] + bias.item_offsets_.loc[5])
    assert preds.loc[-23081] == approx(bias.mean_ + bias.user_offsets_.loc[10])


def test_fallback_save_load(tmp_path):
    tmp_path = lktu.norm_path(tmp_path)

    original = basic.Fallback(basic.Memorized(simple_df), basic.Bias())
    original.fit(lktu.ml_pandas.renamed.ratings)

    fn = tmp_path / 'fallback'
    original.save(fn)

    algo = basic.Fallback(basic.Memorized(simple_df), basic.Bias())
    algo.load(fn)

    bias = algo.algorithms[1]
    assert bias.mean_ == approx(lktu.ml_pandas.ratings.rating.mean())

    # first user + item
    preds = algo.predict_for_user(10, [1])
    assert preds.loc[1] == 4.0
    # second user + first item
    preds = algo.predict_for_user(15, [1])
    assert preds.loc[1] == approx(bias.mean_ + bias.user_offsets_.loc[15] + bias.item_offsets_.loc[1])

    # second item + user item
    preds = algo.predict_for_user(12, [2])
    assert preds.loc[2] == approx(bias.mean_ + bias.user_offsets_.loc[12] + bias.item_offsets_.loc[2])

    # blended
    preds = algo.predict_for_user(10, [1, 5])
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(bias.mean_ + bias.user_offsets_.loc[10] + bias.item_offsets_.loc[5])

    # blended unknown
    preds = algo.predict_for_user(10, [5, 1, -23081])
    assert len(preds) == 3
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(bias.mean_ + bias.user_offsets_.loc[10] + bias.item_offsets_.loc[5])
    assert preds.loc[-23081] == approx(bias.mean_ + bias.user_offsets_.loc[10])


def test_topn_recommend():
    pred = basic.Memorized(simple_df)
    rec = basic.TopN(pred)

    rec10 = rec.recommend(10, candidates=[1, 2])
    assert all(rec10.item == [2, 1])
    assert all(rec10.score == [5, 4])

    rec2 = rec.recommend(12, candidates=[1, 2])
    assert len(rec2) == 1
    assert all(rec2.item == [1])
    assert all(rec2.score == [3])

    rec10 = rec.recommend(10, n=1, candidates=[1, 2])
    assert len(rec10) == 1
    assert all(rec10.item == [2])
    assert all(rec10.score == [5])


def test_popular():
    algo = basic.Popular()
    algo.fit(lktu.ml_pandas.renamed.ratings)
    counts = lktu.ml_pandas.renamed.ratings.groupby('item').user.count()
    counts = counts.nlargest(100)

    assert algo.item_pop_.max() == counts.max()

    recs = algo.recommend(2038, 100)
    assert len(recs) == 100
    assert all(np.diff(recs.score) <= 0)

    assert recs.score.iloc[0] == counts.max()
    # the 10 most popular should be the same
    assert all(counts.index[:10] == recs.item[:10])


def test_pop_candidates():
    algo = basic.Popular()
    algo.fit(lktu.ml_pandas.renamed.ratings)
    counts = lktu.ml_pandas.renamed.ratings.groupby('item').user.count()
    items = lktu.ml_pandas.renamed.ratings.item.unique()

    assert algo.item_pop_.max() == counts.max()

    candidates = np.random.choice(items, 500, replace=False)

    recs = algo.recommend(2038, 100, candidates)
    assert len(recs) == 100
    assert all(np.diff(recs.score) <= 0)

    ccs = counts.loc[candidates]
    ccs = ccs.sort_values(ascending=False)

    assert recs.score.iloc[0] == ccs.max()
    equiv = ccs[ccs == ccs.max()]
    assert recs.item.iloc[0] in equiv.index


def test_pop_save_load(tmp_path):
    tmp_path = lktu.norm_path(tmp_path)
    original = basic.Popular()
    original.fit(lktu.ml_pandas.renamed.ratings)

    fn = tmp_path / 'pop.mod'
    original.save(fn)

    algo = basic.Popular()
    algo.load(fn)

    counts = lktu.ml_pandas.renamed.ratings.groupby('item').user.count()
    counts = counts.nlargest(100)

    assert algo.item_pop_.max() == counts.max()

    recs = algo.recommend(2038, 100)
    assert len(recs) == 100
    assert all(np.diff(recs.score) <= 0)

    assert recs.score.iloc[0] == counts.max()
    # the 10 most popular should be the same
    assert all(counts.index[:10] == recs.item[:10])
