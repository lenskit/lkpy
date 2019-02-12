from lenskit.algorithms import basic
from lenskit import util as lku

import pandas as pd
import numpy as np
import pickle

import lk_test_utils as lktu
from pytest import approx

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


def test_fallback_list():
    algo = basic.Fallback([basic.Memorized(simple_df), basic.Bias()])
    algo.fit(lktu.ml_pandas.renamed.ratings)
    assert len(algo.algorithms) == 2

    params = algo.get_params()
    assert list(params.keys()) == ['algorithms']
    assert len(params['algorithms']) == 2
    assert isinstance(params['algorithms'][0], basic.Memorized)
    assert isinstance(params['algorithms'][1], basic.Bias)


def test_fallback_clone():
    algo = basic.Fallback([basic.Memorized(simple_df), basic.Bias()])
    algo.fit(lktu.ml_pandas.renamed.ratings)
    assert len(algo.algorithms) == 2

    clone = lku.clone(algo)
    assert clone is not algo
    for a1, a2 in zip(algo.algorithms, clone.algorithms):
        assert a1 is not a2
        assert type(a2) == type(a1)


def test_fallback_predict():
    algo = basic.Fallback(basic.Memorized(simple_df), basic.Bias())
    algo.fit(lktu.ml_pandas.renamed.ratings)
    assert len(algo.algorithms) == 2

    bias = algo.algorithms[1]
    assert isinstance(bias, basic.Bias)
    assert bias.mean_ == approx(lktu.ml_pandas.ratings.rating.mean())

    def exp_val(user, item):
        v = bias.mean_
        if user is not None:
            v += bias.user_offsets_.loc[user]
        if item is not None:
            v += bias.item_offsets_.loc[item]
        return v

    # first user + item
    preds = algo.predict_for_user(10, [1])
    assert preds.loc[1] == 4.0
    # second user + first item
    preds = algo.predict_for_user(15, [1])
    assert preds.loc[1] == approx(exp_val(15, 1))

    # second item + user item
    preds = algo.predict_for_user(12, [2])
    assert preds.loc[2] == approx(exp_val(12, 2))

    # blended
    preds = algo.predict_for_user(10, [1, 5])
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(exp_val(10, 5))

    # blended unknown
    preds = algo.predict_for_user(10, [5, 1, -23081])
    assert len(preds) == 3
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(exp_val(10, 5))
    assert preds.loc[-23081] == approx(exp_val(10, None))


def test_fallback_save_load(tmp_path):
    original = basic.Fallback(basic.Memorized(simple_df), basic.Bias())
    original.fit(lktu.ml_pandas.renamed.ratings)

    fn = tmp_path / 'fb.mod'

    with fn.open('wb') as f:
        pickle.dump(original, f)

    with fn.open('rb') as f:
        algo = pickle.load(f)

    bias = algo.algorithms[1]
    assert bias.mean_ == approx(lktu.ml_pandas.ratings.rating.mean())

    def exp_val(user, item):
        v = bias.mean_
        if user is not None:
            v += bias.user_offsets_.loc[user]
        if item is not None:
            v += bias.item_offsets_.loc[item]
        return v

    # first user + item
    preds = algo.predict_for_user(10, [1])
    assert preds.loc[1] == 4.0
    # second user + first item
    preds = algo.predict_for_user(15, [1])
    assert preds.loc[1] == approx(exp_val(15, 1))

    # second item + user item
    preds = algo.predict_for_user(12, [2])
    assert preds.loc[2] == approx(exp_val(12, 2))

    # blended
    preds = algo.predict_for_user(10, [1, 5])
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(exp_val(10, 5))

    # blended unknown
    preds = algo.predict_for_user(10, [5, 1, -23081])
    assert len(preds) == 3
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(exp_val(10, 5))
    assert preds.loc[-23081] == approx(exp_val(10, None))


def test_topn_recommend():
    pred = basic.Memorized(simple_df)
    rec = basic.TopN(pred)
    rec.fit(simple_df)

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


def test_topn_config():
    pred = basic.Memorized(simple_df)
    rec = basic.TopN(pred)

    rs = str(rec)
    assert rs.startswith('TopN/')


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


def test_pop_save_load():
    original = basic.Popular()
    original.fit(lktu.ml_pandas.renamed.ratings)

    mod = pickle.dumps(original)
    algo = pickle.loads(mod)

    counts = lktu.ml_pandas.renamed.ratings.groupby('item').user.count()
    counts = counts.nlargest(100)

    assert algo.item_pop_.max() == counts.max()

    recs = algo.recommend(2038, 100)
    assert len(recs) == 100
    assert all(np.diff(recs.score) <= 0)

    assert recs.score.iloc[0] == counts.max()
    # the 10 most popular should be the same
    assert all(counts.index[:10] == recs.item[:10])


def test_random():
    # test case: no seed
    algo = basic.Random()
    model = algo.train(lktu.ml_pandas.renamed.ratings)
    items = lktu.ml_pandas.renamed.ratings['item'].unique()
    users = lktu.ml_pandas.renamed.ratings['user'].unique()
    nitems = len(items)
    nusers = len(users)

    assert model is not None

    recs1 = algo.recommend(model, 2038, 100)
    recs2 = algo.recommend(model, 2028, 100)
    assert len(recs1) == 100
    assert len(recs2) == 100
    # with very high probabilities
    assert set(recs1['item']) != set(recs2['item'])

    recs_all = algo.recommend(model, 2038)
    assert len(recs_all) == nitems
    assert set(items) == set(recs_all['item'])

    # test case: one single seed
    algo = basic.Random(10)
    model = algo.train(lktu.ml_pandas.renamed.ratings)

    assert model is not None

    # when different users having the same random seed,
    # then recommendations should be the same.
    recs1 = algo.recommend(model, 2038, 100)
    recs2 = algo.recommend(model, 2028, 100)
    assert len(recs1) == 100
    assert set(recs1['item']) == set(recs2['item'])

    recs1 = algo.recommend(model, 2038)
    recs2 = algo.recommend(model, 2028)
    assert len(recs1) == nitems
    assert np.array_equal(recs1['item'].values, recs2['item'].values)

    # test case: seeds Series
    seeds = pd.Series(range(nusers), index=users)
    algo = basic.Random(seeds)
    model1 = algo.train(lktu.ml_pandas.renamed.ratings)
    # same seeds with a second run
    model2 = algo.train(lktu.ml_pandas.renamed.ratings)

    assert model1 is not None
    assert model2 is not None

    # get two different users from all users
    user1, user2 = np.random.choice(users, size=2, replace=False)
    recs1 = algo.recommend(model1, user1, 100)
    recs2 = algo.recommend(model1, user2, 100)
    assert len(recs1) == 100
    assert len(recs2) == 100
    assert set(recs1['item']) != set(recs2['item'])
    recs3 = algo.recommend(model2, user1, 100)
    recs4 = algo.recommend(model2, user2, 100)
    assert len(recs3) == 100
    assert len(recs4) == 100
    assert set(recs3['item']) != set(recs4['item'])
    # consistency between two runs
    assert set(recs1['item']) == set(recs3['item'])
    assert set(recs2['item']) == set(recs4['item'])

    # recommend from candidates
    candidates = np.random.choice(items, 500, replace=False)
    recs1 = algo.recommend(model1, user1, 100, candidates=candidates)
    recs2 = algo.recommend(model1, user2, 100, candidates=candidates)
    assert len(recs1) == 100
    assert len(recs2) == 100
    assert set(recs1['item']) != set(recs2['item'])
    recs3 = algo.recommend(model2, user1, 100, candidates=candidates)
    recs4 = algo.recommend(model2, user2, 100, candidates=candidates)
    assert len(recs3) == 100
    assert len(recs4) == 100
    assert set(recs3['item']) != set(recs4['item'])
    # consistency between two runs
    assert set(recs1['item']) == set(recs3['item'])
    assert set(recs2['item']) == set(recs4['item'])
