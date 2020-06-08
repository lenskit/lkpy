from pathlib import Path
import logging
import numpy as np
import pandas as pd

from pytest import approx, mark

from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms import Recommender
from lenskit.util.test import ml_test
from lenskit.metrics.topn import _dcg
from lenskit import topn, batch, crossfold as xf

_log = logging.getLogger(__name__)


def test_split_keys():
    rla = topn.RecListAnalysis()
    recs, truth = topn._df_keys(['algorithm', 'user', 'item', 'rank', 'score'],
                                ['user', 'item', 'rating'])
    assert truth == ['user']
    assert recs == ['algorithm', 'user']


def test_split_keys_gcol():
    recs, truth = topn._df_keys(['algorithm', 'user', 'item', 'rank', 'score', 'fishtank'],
                                ['user', 'item', 'rating'],
                                ['algorithm', 'fishtank', 'user'])
    assert truth == ['user']
    assert recs == ['algorithm', 'fishtank', 'user']


def test_iter_one():
    df = pd.DataFrame({'user': 1, 'item': [17]})
    gs = list(topn._grouping_iter(df, ['user']))
    assert len(gs) == 1
    uk, idf = gs[0]
    assert uk == (1,)
    assert all(idf['item'] == df['item'])


def test_iter_one_group():
    df = pd.DataFrame({'user': 1, 'item': [17, 13, 24]})
    gs = list(topn._grouping_iter(df, ['user']))
    assert len(gs) == 1
    uk, idf = gs[0]
    assert uk == (1,)
    assert len(idf) == 3
    assert all(idf['item'] == df['item'])


def test_iter_mixed():
    df = pd.DataFrame({'user': [1, 1, 2, 2, 1], 'item': ['a', 'b', 'c', 'd', 'e']})
    gs = list(topn._grouping_iter(df, ['user']))
    assert len(gs) == 2
    uk, idf = gs[0]
    assert uk == (1,)
    assert len(idf) == 3
    assert all(idf['item'] == ['a', 'b', 'e'])

    uk, idf = gs[1]
    assert uk == (2,)
    assert len(idf) == 2
    assert all(idf['item'] == ['c', 'd'])


def test_iter_multilevel():
    df = pd.DataFrame({
        'user': [1, 1, 2, 2, 1],
        'bob': [10, 13, 10, 10, 10],
        'item': ['a', 'b', 'c', 'd', 'e']
    })
    gs = list(topn._grouping_iter(df, ['user', 'bob']))
    assert len(gs) == 3
    uk, idf = gs[0]
    assert uk == (1, 10)
    assert len(idf) == 2
    assert all(idf['item'] == ['a', 'e'])

    uk, idf = gs[1]
    assert uk == (1, 13)
    assert len(idf) == 1
    assert all(idf['item'] == ['b'])

    uk, idf = gs[2]
    assert uk == (2, 10)
    assert len(idf) == 2
    assert all(idf['item'] == ['c', 'd'])


def test_run_one():
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)

    recs = pd.DataFrame({'user': 1, 'item': [2]})
    recs.name = 'recs'
    truth = pd.DataFrame({'user': 1, 'item': [1, 2, 3], 'rating': [3.0, 5.0, 4.0]})
    truth.name = 'truth'

    print(recs)
    print(truth)

    res = rla.compute(recs, truth)
    print(res)

    assert res.index.name == 'user'
    assert res.index.is_unique

    assert len(res) == 1
    assert all(res.index == 1)
    assert all(res.precision == 1.0)
    assert res.recall.values == approx(1/3)


def test_run_two():
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)
    rla.add_metric(topn.ndcg)

    recs = pd.DataFrame({
        'data': 'a',
        'user': ['a', 'a', 'a', 'b', 'b'],
        'item': [2, 3, 1, 4, 5],
        'rank': [1, 2, 3, 1, 2]
    })
    truth = pd.DataFrame({
        'user': ['a', 'a', 'a', 'b', 'b', 'b'],
        'item': [1, 2, 3, 1, 5, 6],
        'rating': [3.0, 5.0, 4.0, 3.0, 5.0, 4.0]
    })

    def prog(inner):
        assert len(inner) == 2
        return inner

    res = rla.compute(recs, truth)
    print(res)

    assert res.columns.nlevels == 1
    assert len(res) == 2
    assert res.index.nlevels == 2
    assert res.index.names == ['data', 'user']
    assert all(res.index.levels[0] == 'a')
    assert all(res.index.levels[1] == ['a', 'b'])
    assert all(res.reset_index().user == ['a', 'b'])
    partial_ndcg = _dcg([0.0, 5.0]) / _dcg([5, 4, 3])
    assert res.ndcg.values == approx([1.0, partial_ndcg])
    assert res.precision.values == approx([1.0, 1/2])
    assert res.recall.values == approx([1.0, 1/3])


def test_inner_format():
    rla = topn.RecListAnalysis()

    recs = pd.DataFrame({
        'data': 'a',
        'user': ['a', 'a', 'a', 'b', 'b'],
        'item': [2, 3, 1, 4, 5],
        'rank': [1, 2, 3, 1, 2]
    })
    truth = pd.DataFrame({
        'user': ['a', 'a', 'a', 'b', 'b', 'b'],
        'item': [1, 2, 3, 1, 5, 6],
        'rating': [3.0, 5.0, 4.0, 3.0, 5.0, 4.0]
    })

    def inner(recs, truth, foo='a'):
        assert foo == 'b'
        assert set(recs.columns) == set(['item', 'rank'])
        assert truth.index.name == 'item'
        assert truth.index.is_unique
        print(truth)
        assert all(truth.columns == ['rating'])
        return len(recs.join(truth, on='item', how='inner'))
    rla.add_metric(inner, name='bob', foo='b')

    res = rla.compute(recs, truth)
    print(res)

    assert len(res) == 2
    assert res.index.nlevels == 2
    assert res.index.names == ['data', 'user']
    assert all(res.index.levels[0] == 'a')
    assert all(res.index.levels[1] == ['a', 'b'])
    assert all(res.reset_index().user == ['a', 'b'])
    assert all(res['bob'] == [3, 1])


def test_spec_group_cols():
    rla = topn.RecListAnalysis(group_cols=['data', 'user'])
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)
    rla.add_metric(topn.ndcg)

    recs = pd.DataFrame({
        'data': 'a',
        'user': ['a', 'a', 'a', 'b', 'b'],
        'item': [2, 3, 1, 4, 5],
        'rank': [1, 2, 3, 1, 2],
        'wombat': np.random.randn(5)
    })
    truth = pd.DataFrame({
        'user': ['a', 'a', 'a', 'b', 'b', 'b'],
        'item': [1, 2, 3, 1, 5, 6],
        'rating': [3.0, 5.0, 4.0, 3.0, 5.0, 4.0]
    })

    res = rla.compute(recs, truth)
    print(res)

    assert len(res) == 2
    assert res.index.nlevels == 2
    assert res.index.names == ['data', 'user']
    assert all(res.index.levels[0] == 'a')
    assert all(res.index.levels[1] == ['a', 'b'])
    assert all(res.reset_index().user == ['a', 'b'])
    partial_ndcg = _dcg([0.0, 5.0]) / _dcg([5, 4, 3])
    assert res.ndcg.values == approx([1.0, partial_ndcg])
    assert res.precision.values == approx([1.0, 1/2])
    assert res.recall.values == approx([1.0, 1/3])


def test_java_equiv():
    dir = Path(__file__).parent
    metrics = pd.read_csv(str(dir / 'topn-java-metrics.csv'))
    recs = pd.read_csv(str(dir / 'topn-java-recs.csv'))
    truth = pd.read_csv(str(dir / 'topn-java-truth.csv'))

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    res = rla.compute(recs, truth)

    umm = pd.merge(metrics, res.reset_index())
    umm['err'] = umm['ndcg'] - umm['Java.nDCG']
    _log.info('merged: \n%s', umm)
    assert umm['err'].values == approx(0, abs=1.0e-6)


@mark.slow
def test_fill_users():
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)

    algo = UserUser(20, min_nbrs=10)
    algo = Recommender.adapt(algo)

    splits = xf.sample_users(ml_test.ratings, 1, 50, xf.SampleN(5))
    train, test = next(splits)
    algo.fit(train)

    rec_users = test['user'].sample(50).unique()
    recs = batch.recommend(algo, rec_users, 25)

    scores = rla.compute(recs, test, include_missing=True)
    assert len(scores) == test['user'].nunique()
    assert scores['recall'].notna().sum() == len(rec_users)
    assert all(scores['ntruth'] == 5)

    mscores = rla.compute(recs, test)
    assert len(mscores) < len(scores)

    recall = scores.loc[scores['recall'].notna(), 'recall'].copy()
    recall, mrecall = recall.align(mscores['recall'])
    assert all(recall == mrecall)


@mark.slow
def test_adv_fill_users():
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)

    a_uu = UserUser(30, min_nbrs=10)
    a_uu = Recommender.adapt(a_uu)
    a_ii = ItemItem(20, min_nbrs=4)
    a_ii = Recommender.adapt(a_ii)

    splits = xf.sample_users(ml_test.ratings, 2, 50, xf.SampleN(5))
    all_recs = {}
    all_test = {}
    for i, (train, test) in enumerate(splits):
        a_uu.fit(train)
        rec_users = test['user'].sample(50).unique()
        all_recs[(i+1, 'UU')] = batch.recommend(a_uu, rec_users, 25)

        a_ii.fit(train)
        rec_users = test['user'].sample(50).unique()
        all_recs[(i+1, 'II')] = batch.recommend(a_ii, rec_users, 25)
        all_test[i+1] = test

    recs = pd.concat(all_recs, names=['part', 'algo'])
    recs.reset_index(['part', 'algo'], inplace=True)
    recs.reset_index(drop=True, inplace=True)

    test = pd.concat(all_test, names=['part'])
    test.reset_index(['part'], inplace=True)
    test.reset_index(drop=True, inplace=True)

    scores = rla.compute(recs, test, include_missing=True)
    inames = scores.index.names
    scores.sort_index(inplace=True)
    assert len(scores) == 50 * 4
    assert all(scores['ntruth'] == 5)
    assert scores['recall'].isna().sum() > 0
    _log.info('scores:\n%s', scores)

    mscores = rla.compute(recs, test)
    mscores = mscores.reset_index().set_index(inames)
    mscores.sort_index(inplace=True)
    assert len(mscores) < len(scores)
    _log.info('mscores:\n%s', mscores)

    recall = scores.loc[scores['recall'].notna(), 'recall'].copy()
    recall, mrecall = recall.align(mscores['recall'])
    assert all(recall == mrecall)
