# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import linalg as la

import pytest
from pytest import approx, fixture, mark

import lenskit.algorithms.knn.item as knn
import lenskit.util.test as lktu
from lenskit import batch
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Fallback
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.ranking import TopN
from lenskit.data.dataset import from_interactions_df
from lenskit.data.vocab import EntityId, Vocabulary
from lenskit.diagnostics import ConfigWarning, DataWarning
from lenskit.util import clone
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401

_log = logging.getLogger(__name__)

simple_ratings = pd.DataFrame.from_records(
    [
        (1, 6, 4.0),
        (2, 6, 2.0),
        (1, 7, 3.0),
        (2, 7, 2.0),
        (3, 7, 5.0),
        (4, 7, 2.0),
        (1, 8, 3.0),
        (2, 8, 4.0),
        (3, 8, 3.0),
        (4, 8, 2.0),
        (5, 8, 3.0),
        (6, 8, 2.0),
        (1, 9, 3.0),
        (3, 9, 4.0),
    ],
    columns=["user", "item", "rating"],
)
simple_ds = from_interactions_df(simple_ratings)


@fixture(scope="module")
def ml_subset(ml_ratings):
    "Fixture that returns a subset of the MovieLens database."
    icounts = ml_ratings.groupby("item").rating.count()
    top = icounts.nlargest(500)
    top_rates = ml_ratings[ml_ratings["item"].isin(top.index)]
    _log.info("top 500 items yield %d of %d ratings", len(top_rates), len(ml_ratings))
    return top_rates


def test_ii_dft_config():
    algo = knn.ItemItem(30, save_nbrs=500)
    assert algo.center
    assert algo.aggregate == "weighted-average"
    assert algo.use_ratings


def test_ii_exp_config():
    algo = knn.ItemItem(30, save_nbrs=500, feedback="explicit")
    assert algo.center
    assert algo.aggregate == "weighted-average"
    assert algo.use_ratings


def test_ii_imp_config():
    algo = knn.ItemItem(30, save_nbrs=500, feedback="implicit")
    assert not algo.center
    assert algo.aggregate == "sum"
    assert not algo.use_ratings


def test_ii_imp_clone():
    algo = knn.ItemItem(30, save_nbrs=500, feedback="implicit")
    a2 = clone(algo)

    assert a2.get_params() == algo.get_params()
    assert a2.__dict__ == algo.__dict__


def test_ii_train():
    algo = knn.ItemItem(30, save_nbrs=500)
    algo.fit(simple_ds)

    assert isinstance(algo.item_means_, torch.Tensor)
    assert isinstance(algo.item_counts_, torch.Tensor)
    matrix = algo.sim_matrix_

    test_means = simple_ratings.groupby("item")["rating"].mean()
    test_means = test_means.reindex(algo.items_.ids())
    assert np.all(algo.item_means_.numpy() == test_means.values.astype("f8"))

    # 6 is a neighbor of 7
    six, seven = algo.items_.numbers([6, 7])
    _log.info("six: %d", six)
    _log.info("seven: %d", seven)
    _log.info("matrix: %s", algo.sim_matrix_)
    assert matrix[six, seven] > 0
    # and has the correct score
    six_v = simple_ratings[simple_ratings.item == 6].set_index("user").rating
    six_v = six_v - six_v.mean()
    seven_v = simple_ratings[simple_ratings.item == 7].set_index("user").rating
    seven_v = seven_v - seven_v.mean()
    denom = la.norm(six_v.values) * la.norm(seven_v.values)
    six_v, seven_v = six_v.align(seven_v, join="inner")
    num = six_v.dot(seven_v)
    assert matrix[six, seven] == approx(num / denom, 0.01)  # type: ignore

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values().numpy())))
    assert all(algo.sim_matrix_.values() > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values() < 1 + 1.0e-6)


def test_ii_train_unbounded():
    algo = knn.ItemItem(30)
    algo.fit(simple_ds)

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values())))
    assert all(algo.sim_matrix_.values() > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values() < 1 + 1.0e-6)

    # 6 is a neighbor of 7
    matrix = algo.sim_matrix_
    six, seven = algo.items_.numbers([6, 7])
    assert matrix[six, seven] > 0

    # and has the correct score
    six_v = simple_ratings[simple_ratings.item == 6].set_index("user").rating
    six_v = six_v - six_v.mean()
    seven_v = simple_ratings[simple_ratings.item == 7].set_index("user").rating
    seven_v = seven_v - seven_v.mean()
    denom = la.norm(six_v.values) * la.norm(seven_v.values)
    six_v, seven_v = six_v.align(seven_v, join="inner")
    num = six_v.dot(seven_v)
    assert matrix[six, seven] == approx(num / denom, 0.01)  # type: ignore


def test_ii_simple_predict():
    algo = knn.ItemItem(30, save_nbrs=500)
    algo.fit(simple_ds)

    res = algo.predict_for_user(3, [6])
    _log.info("got predictions: %s", res)
    assert res is not None
    assert len(res) == 1
    assert 6 in res.index
    assert not np.isnan(res.loc[6])


def test_ii_simple_implicit_predict():
    algo = knn.ItemItem(30, center=False, aggregate="sum")
    algo.fit(from_interactions_df(simple_ratings.loc[:, ["user", "item"]]))

    res = algo.predict_for_user(3, [6])
    assert res is not None
    assert len(res) == 1
    assert 6 in res.index
    assert not np.isnan(res.loc[6])
    assert res.loc[6] > 0


def test_ii_warns_center():
    "Test that item-item warns if you center non-centerable data"
    data = simple_ratings.assign(rating=1)
    algo = knn.ItemItem(5)
    with pytest.warns(DataWarning):
        algo.fit(from_interactions_df(data))


def test_ii_warns_center_with_no_use_ratings():
    "Test that item-item warns if you configure to ignore ratings but center."
    with pytest.warns(ConfigWarning):
        knn.ItemItem(5, use_ratings=False, aggregate="sum")


def test_ii_warns_wa_with_no_use_ratings():
    "Test that item-item warns if you configure to ignore ratings but use weighted-average."
    with pytest.warns(ConfigWarning):
        algo = knn.ItemItem(5, use_ratings=False, center=False)
        assert not algo.use_ratings
        assert not algo.center
        assert algo.aggregate == algo.AGG_WA


@lktu.wantjit
@mark.slow
def test_ii_train_ml100k(tmp_path, ml_100k):
    "Test an unbounded model on ML-100K"
    algo = knn.ItemItem(30)
    _log.info("training model")
    algo.fit(from_interactions_df(ml_100k))

    _log.info("testing model")

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values())))
    assert all(algo.sim_matrix_.values() > 0)

    # a little tolerance
    assert np.max(algo.sim_matrix_.values().numpy()) <= 1

    assert algo.item_counts_.sum() == len(algo.sim_matrix_.values())

    means = ml_100k.groupby("item").rating.mean()
    assert means[algo.items_.ids()].values == approx(algo.item_means_)

    # save
    fn = tmp_path / "ii.mod"
    _log.info("saving model to %s", fn)
    with fn.open("wb") as modf:
        pickle.dump(algo, modf)

    _log.info("reloading model")
    with fn.open("rb") as modf:
        restored = pickle.load(modf)

    assert all(restored.sim_matrix_.values() > 0)

    r_mat = restored.sim_matrix_
    o_mat = algo.sim_matrix_

    assert all(r_mat.values() == o_mat.values())


@lktu.wantjit
@mark.slow
def test_ii_large_models(rng, ml_ratings, ml_ds):
    "Several tests of large trained I-I models"
    _log.info("training limited model")
    MODEL_SIZE = 100
    algo_lim = knn.ItemItem(30, save_nbrs=MODEL_SIZE)
    algo_lim.fit(ml_ds)

    _log.info("training unbounded model")
    algo_ub = knn.ItemItem(30)
    algo_ub.fit(ml_ds)

    _log.info("testing models")
    assert all(np.logical_not(np.isnan(algo_lim.sim_matrix_.values())))
    assert algo_lim.sim_matrix_.values().min() > 0
    # a little tolerance
    assert algo_lim.sim_matrix_.values().max() <= 1

    means = ml_ratings.groupby("item").rating.mean()
    assert means[algo_lim.items_.ids()].values == approx(algo_lim.item_means_)

    assert all(np.logical_not(np.isnan(algo_ub.sim_matrix_.values())))
    assert algo_ub.sim_matrix_.values().min() > 0
    assert algo_ub.sim_matrix_.values().max() <= 1

    means = ml_ratings.groupby("item").rating.mean()
    assert means[algo_ub.items_.ids()].values == approx(algo_ub.item_means_)

    mc_rates = (
        ml_ratings.set_index("item")
        .join(pd.DataFrame({"item_mean": means}))
        .assign(rating=lambda df: df.rating - df.item_mean)
    )

    mat_lim = algo_lim.sim_matrix_
    mat_ub = algo_ub.sim_matrix_

    _log.info("make sure the similarity matrix is sorted")
    for i in range(algo_lim.items_.size):
        sp = algo_lim.sim_matrix_.crow_indices()[i]
        ep = algo_lim.sim_matrix_.crow_indices()[i + 1]
        cols = algo_lim.sim_matrix_.col_indices()[sp:ep]
        diffs = np.diff(cols.numpy())
        if np.any(diffs <= 0):
            _log.error("row %d: %d non-sorted indices", i, np.sum(diffs <= 0))
            (bad,) = np.nonzero(diffs <= 0)
            for i in bad:
                _log.info("bad indices %d: %d %d", i, cols[i], cols[i + 1])
            raise AssertionError(f"{np.sum(diffs <= 0)} non-sorted indices")

    _log.info("checking a sample of neighborhoods")
    items = algo_ub.items_.ids()
    items = items[algo_ub.item_counts_.numpy() > 0]
    for i in rng.choice(items, 50):
        ipos = algo_ub.items_.number(i)
        _log.debug("checking item %d at position %d", i, ipos)
        assert ipos == algo_lim.items_.number(i)
        irates = mc_rates.loc[[i], :].set_index("user").rating

        ub_row = mat_ub[ipos]
        b_row = mat_lim[ipos]
        assert len(b_row.values()) <= MODEL_SIZE
        ub_cols = ub_row.indices()[0].numpy()
        b_cols = b_row.indices()[0].numpy()
        _log.debug("kept %d of %d neighbors", len(b_cols), len(ub_cols))

        _log.debug("checking for sorted indices")
        assert np.all(np.diff(ub_cols) > 0)
        assert np.all(np.diff(b_cols) > 0)

        # all bounded columns are in the unbounded columns
        _log.debug("checking that bounded columns are a subset of unbounded")
        present = np.isin(b_cols, ub_cols)
        if not np.all(present):
            _log.error("missing items: %s", b_cols[~present])
            _log.error("scores: %s", b_row.values()[~present])  # type: ignore
            raise AssertionError(f"missing {np.sum(~present)} values from unbounded")

        # spot-check some similarities
        _log.debug("checking equal similarities")
        for n in rng.choice(ub_cols, min(10, len(ub_cols))):
            n_id = algo_ub.items_.id(n)
            n_rates = mc_rates.loc[n_id, :].set_index("user").rating
            ir, nr = irates.align(n_rates, fill_value=0)
            cor = ir.corr(nr)
            assert mat_ub[ipos, n].item() == approx(cor, abs=1.0e-6)

        # short rows are equal
        if len(b_cols) < MODEL_SIZE:
            _log.debug("short row of length %d", len(b_cols))
            assert len(b_row) == len(ub_row)
            assert b_row.values().numpy() == approx(ub_row.values().numpy())
            continue

        # row is truncated - check that truncation is correct
        ub_nbrs = pd.Series(ub_row.values().numpy(), algo_ub.items_.ids(ub_cols))
        b_nbrs = pd.Series(b_row.values().numpy(), algo_lim.items_.ids(b_cols))

        assert len(ub_nbrs) >= len(b_nbrs)
        assert len(b_nbrs) <= MODEL_SIZE
        assert all(b_nbrs.index.isin(ub_nbrs.index))
        # the similarities should be equal!
        b_match, ub_match = b_nbrs.align(ub_nbrs, join="inner")
        assert all(b_match == b_nbrs)
        assert b_match.values == approx(ub_match.values)
        assert b_nbrs.max() == approx(ub_nbrs.max())
        if len(ub_nbrs) > MODEL_SIZE:
            assert len(b_nbrs) == MODEL_SIZE
            ub_shrink = ub_nbrs.nlargest(MODEL_SIZE)
            # the minimums should be equal
            assert ub_shrink.min() == approx(b_nbrs.min())
            # everything above minimum value should be the same set of items
            # the minimum value might be a tie
            ubs_except_min = ub_shrink[ub_shrink > b_nbrs.min()]
            missing = ~ubs_except_min.index.isin(b_nbrs.index)
            if np.any(missing):
                _log.error("missing unbounded values:\n%s", ubs_except_min[missing])
                raise AssertionError(f"missing {np.sum(missing)} unbounded values")


@lktu.wantjit
@mark.slow
def test_ii_implicit_large(rng, ml_ratings):
    "Test that implicit-feedback mode works on full test data."
    _log.info("training model")
    NBRS = 5
    NUSERS = 25
    NRECS = 50
    algo = knn.ItemItem(NBRS, feedback="implicit")
    _log.info("agg: %s", algo.aggregate)
    algo = Recommender.adapt(algo)
    algo.fit(from_interactions_df(ml_ratings[["user", "item"]], item_col="item"))
    assert isinstance(algo, TopN)

    users = rng.choice(ml_ratings["user"].unique(), NUSERS)

    items: Vocabulary = algo.predictor.items_
    mat: torch.Tensor = algo.predictor.sim_matrix_.to_dense()

    for user in users:
        recs = algo.recommend(user, NRECS)
        _log.info("user %s recs\n%s", user, recs)
        assert len(recs) == NRECS
        urates = ml_ratings[ml_ratings["user"] == user]

        smat = mat[torch.from_numpy(items.numbers(urates["item"].values)), :]
        for row in recs.itertuples():
            col = smat[:, items.number(row.item)]
            top, _is = torch.topk(col, NBRS)
            score = top.sum()
            try:
                assert row.score == approx(score)
            except AssertionError as e:
                _log.error("test failed for user %s item %s", user, row.item)
                _log.info("score: %.6f", row.score)
                _log.info("sims:\n%s", col)
                _log.info("total: %.3f", col.sum())
                _log.info("filtered: %s", top)
                _log.info("filtered sum: %.3f", top.sum())
                raise e


@lktu.wantjit
def test_ii_save_load(tmp_path, ml_ratings, ml_subset):
    "Save and load a model"
    original = knn.ItemItem(30, save_nbrs=500)
    _log.info("building model")
    original.fit(from_interactions_df(ml_subset, item_col="item"))

    fn = tmp_path / "ii.mod"
    _log.info("saving model to %s", fn)
    with fn.open("wb") as modf:
        pickle.dump(original, modf)

    _log.info("pickled %d bytes", fn.stat().st_size)
    _log.info("reloading model")
    with fn.open("rb") as modf:
        algo = pickle.load(modf)

    _log.info("checking model")
    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values())))
    assert all(algo.sim_matrix_.values() > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values() < 1 + 1.0e-6)

    assert all(algo.item_counts_ == original.item_counts_)
    assert algo.item_counts_.sum() == len(algo.sim_matrix_.values())
    assert len(algo.sim_matrix_.values()) == len(algo.sim_matrix_.values())
    assert all(algo.sim_matrix_.crow_indices() == original.sim_matrix_.crow_indices())
    assert algo.sim_matrix_.values() == approx(original.sim_matrix_.values())

    r_mat = algo.sim_matrix_
    o_mat = original.sim_matrix_
    assert all(r_mat.crow_indices() == o_mat.crow_indices())

    means = ml_ratings.groupby("item").rating.mean()
    assert means[algo.items_.ids()].values == approx(original.item_means_)


@lktu.wantjit
def test_ii_implicit_save_load(tmp_path, ml_subset):
    "Save and load a model"
    original = knn.ItemItem(30, save_nbrs=500, center=False, aggregate="sum")
    _log.info("building model")
    original.fit(from_interactions_df(ml_subset.loc[:, ["user", "item"]], item_col="item"))

    fn = tmp_path / "ii.mod"
    _log.info("saving model to %s", fn)
    with fn.open("wb") as modf:
        pickle.dump(original, modf)
    _log.info("pickled %d bytes", fn.stat().st_size)

    _log.info("reloading model")
    with fn.open("rb") as modf:
        algo = pickle.load(modf)

    _log.info("checking model")
    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values())))
    assert all(algo.sim_matrix_.values() > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values() < 1 + 1.0e-6)

    assert all(algo.item_counts_ == original.item_counts_)
    assert algo.item_counts_.sum() == len(algo.sim_matrix_.values())
    assert algo.sim_matrix_.values().shape == original.sim_matrix_.values().shape
    assert all(algo.sim_matrix_.crow_indices() == original.sim_matrix_.crow_indices())
    assert algo.sim_matrix_.values() == approx(original.sim_matrix_.values())

    assert algo.item_means_ is None


@lktu.wantjit
@mark.slow
def test_ii_old_implicit(ml_ratings):
    algo = knn.ItemItem(20, save_nbrs=100, center=False, aggregate="sum")
    data = ml_ratings.loc[:, ["user", "item"]]

    algo.fit(from_interactions_df(data, item_col="item"))
    assert algo.item_counts_.sum() == algo.sim_matrix_.values().shape[0]
    assert all(algo.sim_matrix_.values() > 0)
    assert all(algo.item_counts_ <= 100)

    preds = algo.predict_for_user(50, [1, 2, 42])
    assert all(preds[preds.notna()] > 0)


@lktu.wantjit
@mark.slow
def test_ii_no_ratings(ml_ratings, ml_ds):
    a1 = knn.ItemItem(20, save_nbrs=100, center=False, aggregate="sum")
    a1.fit(from_interactions_df(ml_ratings.loc[:, ["user", "item"]], item_col="item"))

    algo = knn.ItemItem(20, save_nbrs=100, feedback="implicit")

    algo.fit(ml_ds)
    assert algo.item_counts_.sum().item() == algo.sim_matrix_.values().shape[0]
    assert all(algo.sim_matrix_.values() > 0)
    assert all(algo.item_counts_ <= 100)

    preds = algo.predict_for_user(50, [1, 2, 42])
    assert all(preds[preds.notna()] > 0)
    p2 = algo.predict_for_user(50, [1, 2, 42])
    preds, p2 = preds.align(p2)
    assert preds.values == approx(p2.values, nan_ok=True)


@mark.slow
@mark.eval
def test_ii_batch_accuracy(ml_100k):
    import lenskit.crossfold as xf
    import lenskit.metrics.predict as pm
    from lenskit import batch
    from lenskit.algorithms import basic, bias

    ii_algo = knn.ItemItem(30)
    algo = basic.Fallback(ii_algo, bias.Bias())

    def eval(train, test):
        _log.info("running training")
        algo.fit(from_interactions_df(train))
        _log.info("testing %d users", test.user.nunique())
        return batch.predict(algo, test, n_jobs=1)

    preds = pd.concat(
        (eval(train, test) for (train, test) in xf.partition_users(ml_100k, 5, xf.SampleFrac(0.2)))
    )
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.70, abs=0.025)

    user_rmse = preds.groupby("user").apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.90, abs=0.05)


@lktu.wantjit
@mark.slow
def test_ii_known_preds(ml_ds):
    from lenskit import batch

    algo = knn.ItemItem(20, min_sim=1.0e-6)
    _log.info("training %s on ml data", algo)
    algo.fit(ml_ds)
    assert algo.center
    assert algo.item_means_ is not None
    _log.info("model means: %s", algo.item_means_)

    dir = Path(__file__).parent
    pred_file = dir / "item-item-preds.csv"
    _log.info("reading known predictions from %s", pred_file)
    known_preds = pd.read_csv(str(pred_file))
    pairs = known_preds.loc[:, ["user", "item"]]

    preds = batch.predict(algo, pairs)
    merged = pd.merge(known_preds.rename(columns={"prediction": "expected"}), preds)
    assert len(merged) == len(preds)
    merged["error"] = merged.expected - merged.prediction
    try:
        assert not any(merged.prediction.isna() & merged.expected.notna())
    except AssertionError as e:
        bad = merged[merged.prediction.isna() & merged.expected.notna()]
        _log.error("erroneously missing or present predictions:\n%s", bad)
        raise e

    err = merged.error
    err = err[err.notna()]
    space = np.zeros(7)
    space[1:] = np.logspace(-6, -1, 6)
    counts, edges = np.histogram(np.abs(err), space)
    _log.info("error histogram: %s", counts)
    try:
        # no more than 5 are out-of-bounds
        assert np.sum(space[1:]) < 5
    except AssertionError as e:
        bad = merged[merged.error.notna() & (merged.error.abs() >= 0.01)]
        _log.error("erroneous predictions:\n%s", bad)
        raise e


@lktu.wantjit
@mark.slow
@mark.eval
@mark.parametrize("ncpus", [1, 2])
def test_ii_batch_recommend(ml_100k, ncpus):
    import lenskit.crossfold as xf
    from lenskit import topn

    def eval(train, test):
        _log.info("running training")
        algo = knn.ItemItem(30)
        algo = Recommender.adapt(algo)
        algo.fit(from_interactions_df(train))
        _log.info("testing %d users", test.user.nunique())
        recs = batch.recommend(algo, test.user.unique(), 100, n_jobs=ncpus)
        return recs

    test_frames = []
    recs = []
    for train, test in xf.partition_users(ml_100k, 5, xf.SampleFrac(0.2)):
        test_frames.append(test)
        recs.append(eval(train, test))

    test = pd.concat(test_frames)
    recs = pd.concat(recs)

    _log.info("analyzing recommendations")
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, test)
    dcg = results.ndcg
    _log.info("nDCG for %d users is %f", len(dcg), dcg.mean())
    assert dcg.mean() > 0.03
