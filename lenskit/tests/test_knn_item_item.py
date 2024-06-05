# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import gc
import logging
import pickle
from pathlib import Path

import csr.kernel as csrk
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
from lenskit.diagnostics import ConfigWarning, DataWarning
from lenskit.parallel import invoker
from lenskit.util import Stopwatch, clone

_log = logging.getLogger(__name__)

ml_ratings = lktu.ml_test.ratings
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


@fixture(scope="module")
def ml_subset():
    "Fixture that returns a subset of the MovieLens database."
    ratings = lktu.ml_test.ratings
    icounts = ratings.groupby("item").rating.count()
    top = icounts.nlargest(500)
    ratings = ratings.set_index("item")
    top_rates = ratings.loc[top.index, :]
    _log.info("top 500 items yield %d of %d ratings", len(top_rates), len(ratings))
    return top_rates.reset_index()


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
    algo.fit(simple_ratings)

    assert isinstance(algo.item_index_, pd.Index)
    assert isinstance(algo.item_means_, torch.Tensor)
    assert isinstance(algo.item_counts_, torch.Tensor)
    matrix = algo.sim_matrix_

    test_means = simple_ratings.groupby("item")["rating"].mean()
    test_means = test_means.reindex(algo.item_index_)
    assert np.all(algo.item_means_.numpy() == test_means.values.astype("f8"))

    # 6 is a neighbor of 7
    six, seven = algo.item_index_.get_indexer([6, 7])
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
    assert matrix[six, seven] == approx(num / denom, 0.01)

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values().numpy())))
    assert all(algo.sim_matrix_.values() > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values() < 1 + 1.0e-6)


def test_ii_train_unbounded():
    algo = knn.ItemItem(30)
    algo.fit(simple_ratings)

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values())))
    assert all(algo.sim_matrix_.values() > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values() < 1 + 1.0e-6)

    # 6 is a neighbor of 7
    matrix = algo.sim_matrix_
    six, seven = algo.item_index_.get_indexer([6, 7])
    assert matrix[six, seven] > 0

    # and has the correct score
    six_v = simple_ratings[simple_ratings.item == 6].set_index("user").rating
    six_v = six_v - six_v.mean()
    seven_v = simple_ratings[simple_ratings.item == 7].set_index("user").rating
    seven_v = seven_v - seven_v.mean()
    denom = la.norm(six_v.values) * la.norm(seven_v.values)
    six_v, seven_v = six_v.align(seven_v, join="inner")
    num = six_v.dot(seven_v)
    assert matrix[six, seven] == approx(num / denom, 0.01)


def test_ii_simple_predict():
    algo = knn.ItemItem(30, save_nbrs=500)
    algo.fit(simple_ratings)

    res = algo.predict_for_user(3, [6])
    _log.info("got predictions: %s", res)
    assert res is not None
    assert len(res) == 1
    assert 6 in res.index
    assert not np.isnan(res.loc[6])


def test_ii_simple_implicit_predict():
    algo = knn.ItemItem(30, center=False, aggregate="sum")
    algo.fit(simple_ratings.loc[:, ["user", "item"]])

    res = algo.predict_for_user(3, [6])
    assert res is not None
    assert len(res) == 1
    assert 6 in res.index
    assert not np.isnan(res.loc[6])
    assert res.loc[6] > 0


@mark.skip("currently broken")
def test_ii_warn_duplicates():
    extra = pd.DataFrame.from_records([(3, 7, 4.5)], columns=["user", "item", "rating"])
    ratings = pd.concat([simple_ratings, extra])
    algo = knn.ItemItem(5)
    algo.fit(ratings)

    try:
        with pytest.warns(DataWarning):
            algo.predict_for_user(3, [6])
    except AssertionError:
        pass  # this is fine


def test_ii_warns_center():
    "Test that item-item warns if you center non-centerable data"
    data = simple_ratings.assign(rating=1)
    algo = knn.ItemItem(5)
    with pytest.warns(DataWarning):
        algo.fit(data)


def test_ii_warns_center_with_no_use_ratings():
    "Test that item-item warns if you configure to ignore ratings but center."
    with pytest.warns(ConfigWarning):
        knn.ItemItem(5, use_ratings=False, aggregate="sum")


def test_ii_warns_wa_with_no_use_ratings():
    "Test that item-item warns if you configure to ignore ratings but weighted=average."
    with pytest.warns(ConfigWarning):
        algo = knn.ItemItem(5, use_ratings=False, center=False)


@lktu.wantjit
@mark.skip("redundant with large_models")
def test_ii_train_big():
    "Simple tests for bounded models"
    algo = knn.ItemItem(30, save_nbrs=500)
    algo.fit(ml_ratings)

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values())))
    assert all(algo.sim_matrix_.values() > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values() < 1 + 1.0e-6)

    assert algo.item_counts_.sum() == algo.sim_matrix_.nnz

    means = ml_ratings.groupby("item").rating.mean()
    assert means[algo.item_index_].values == approx(algo.item_means_)


@lktu.wantjit
@mark.skip("redundant with large_models")
def test_ii_train_big_unbounded():
    "Simple tests for unbounded models"
    algo = knn.ItemItem(30)
    algo.fit(ml_ratings)

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values())))
    assert all(algo.sim_matrix_.values() > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values() < 1 + 1.0e-6)

    assert algo.item_counts_.sum() == algo.sim_matrix_.nnz

    means = ml_ratings.groupby("item").rating.mean()
    assert means[algo.item_index_].values == approx(algo.item_means_)


@lktu.wantjit
@mark.skipif(not lktu.ml100k.available, reason="ML100K data not present")
def test_ii_train_ml100k(tmp_path):
    "Test an unbounded model on ML-100K"
    ratings = lktu.ml100k.ratings
    algo = knn.ItemItem(30)
    _log.info("training model")
    algo.fit(ratings)

    _log.info("testing model")

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values())))
    assert all(algo.sim_matrix_.values() > 0)

    # a little tolerance
    assert np.max(algo.sim_matrix_.values().numpy()) <= 1

    assert algo.item_counts_.sum() == len(algo.sim_matrix_.values())

    means = ratings.groupby("item").rating.mean()
    assert means[algo.item_index_].values == approx(algo.item_means_)

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
def test_ii_large_models(rng):
    "Several tests of large trained I-I models"
    _log.info("training limited model")
    MODEL_SIZE = 100
    algo_lim = knn.ItemItem(30, save_nbrs=MODEL_SIZE)
    algo_lim.fit(ml_ratings)

    _log.info("training unbounded model")
    algo_ub = knn.ItemItem(30)
    algo_ub.fit(ml_ratings)

    _log.info("testing models")
    assert all(np.logical_not(np.isnan(algo_lim.sim_matrix_.values())))
    assert algo_lim.sim_matrix_.values().min() > 0
    # a little tolerance
    assert algo_lim.sim_matrix_.values().max() <= 1

    means = ml_ratings.groupby("item").rating.mean()
    assert means[algo_lim.item_index_].values == approx(algo_lim.item_means_)

    assert all(np.logical_not(np.isnan(algo_ub.sim_matrix_.values())))
    assert algo_ub.sim_matrix_.values().min() > 0
    assert algo_ub.sim_matrix_.values().max() <= 1

    means = ml_ratings.groupby("item").rating.mean()
    assert means[algo_ub.item_index_].values == approx(algo_ub.item_means_)

    mc_rates = (
        ml_ratings.set_index("item")
        .join(pd.DataFrame({"item_mean": means}))
        .assign(rating=lambda df: df.rating - df.item_mean)
    )

    mat_lim = algo_lim.sim_matrix_
    mat_ub = algo_ub.sim_matrix_

    _log.info("make sure the similarity matrix is sorted")
    for i in range(len(algo_lim.item_index_)):
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
    items = algo_ub.item_index_.values
    items = items[algo_ub.item_counts_.numpy() > 0]
    for i in rng.choice(items, 50):
        ipos = algo_ub.item_index_.get_loc(i)
        _log.debug("checking item %d at position %d", i, ipos)
        assert ipos == algo_lim.item_index_.get_loc(i)
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
            _log.error("scores: %s", b_row.values()[~present])
            raise AssertionError(f"missing {np.sum(~present)} values from unbounded")

        # spot-check some similarities
        _log.debug("checking equal similarities")
        for n in rng.choice(ub_cols, min(10, len(ub_cols))):
            n_id = algo_ub.item_index_[n]
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
        ub_nbrs = pd.Series(ub_row.values().numpy(), algo_ub.item_index_[ub_cols])
        b_nbrs = pd.Series(b_row.values().numpy(), algo_lim.item_index_[b_cols])

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
def test_ii_implicit_large(rng):
    "Test that implicit-feedback mode works on full test data."
    _log.info("training model")
    NBRS = 5
    NUSERS = 25
    NRECS = 50
    algo = knn.ItemItem(NBRS, feedback="implicit")
    _log.info("agg: %s", algo.aggregate)
    algo = Recommender.adapt(algo)
    algo.fit(ml_ratings[["user", "item"]])

    users = rng.choice(ml_ratings["user"].unique(), NUSERS)

    items: pd.Index = algo.predictor.item_index_
    mat: torch.Tensor = algo.predictor.sim_matrix_.to_dense()

    for user in users:
        recs = algo.recommend(user, NRECS)
        _log.info("user %s recs\n%s", user, recs)
        assert len(recs) == NRECS
        urates = ml_ratings[ml_ratings["user"] == user]

        smat = mat[torch.from_numpy(items.get_indexer_for(urates["item"].values)), :]
        for row in recs.itertuples():
            col = smat[:, items.get_loc(row.item)]
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
def test_ii_save_load(tmp_path, ml_subset):
    "Save and load a model"
    original = knn.ItemItem(30, save_nbrs=500)
    _log.info("building model")
    original.fit(ml_subset)

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
    assert means[algo.item_index_].values == approx(original.item_means_)


@lktu.wantjit
def test_ii_implicit_save_load(tmp_path, ml_subset):
    "Save and load a model"
    original = knn.ItemItem(30, save_nbrs=500, center=False, aggregate="sum")
    _log.info("building model")
    original.fit(ml_subset.loc[:, ["user", "item"]])

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
def test_ii_old_implicit():
    algo = knn.ItemItem(20, save_nbrs=100, center=False, aggregate="sum")
    data = ml_ratings.loc[:, ["user", "item"]]

    algo.fit(data)
    assert algo.item_counts_.sum() == algo.sim_matrix_.values().shape[0]
    assert all(algo.sim_matrix_.values() > 0)
    assert all(algo.item_counts_ <= 100)

    preds = algo.predict_for_user(50, [1, 2, 42])
    assert all(preds[preds.notna()] > 0)


@lktu.wantjit
@mark.slow
def test_ii_no_ratings():
    a1 = knn.ItemItem(20, save_nbrs=100, center=False, aggregate="sum")
    a1.fit(ml_ratings.loc[:, ["user", "item"]])

    algo = knn.ItemItem(20, save_nbrs=100, feedback="implicit")

    algo.fit(ml_ratings)
    assert algo.item_counts_.sum().item() == algo.sim_matrix_.values().shape[0]
    assert all(algo.sim_matrix_.values() > 0)
    assert all(algo.item_counts_ <= 100)

    preds = algo.predict_for_user(50, [1, 2, 42])
    assert all(preds[preds.notna()] > 0)
    p2 = algo.predict_for_user(50, [1, 2, 42])
    preds, p2 = preds.align(p2)
    assert preds.values == approx(p2.values, nan_ok=True)


@mark.slow
@pytest.mark.skip("fast/slow paths have been removed")
def test_ii_implicit_fast_ident():
    algo = knn.ItemItem(20, save_nbrs=100, center=False, aggregate="sum")
    data = ml_ratings.loc[:, ["user", "item"]]

    algo.fit(data)
    assert algo.item_counts_.sum() == algo.sim_matrix_.values().shape
    assert all(algo.sim_matrix_.values() > 0)
    assert all(algo.item_counts_ <= 100)

    preds = algo.predict_for_user(50, [1, 2, 42])
    assert all(preds[preds.notna()] > 0)
    assert np.isnan(preds.iloc[2])

    algo.min_sim = -1  # force it to take the slow path for all predictions
    p2 = algo.predict_for_user(50, [1, 2, 42])
    assert preds.values[:2] == approx(p2.values[:2])
    assert np.isnan(p2.iloc[2])


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason="ML100K data not present")
def test_ii_batch_accuracy():
    import lenskit.crossfold as xf
    import lenskit.metrics.predict as pm
    from lenskit import batch
    from lenskit.algorithms import basic, bias

    ratings = lktu.ml100k.ratings

    ii_algo = knn.ItemItem(30)
    algo = basic.Fallback(ii_algo, bias.Bias())

    def eval(train, test):
        _log.info("running training")
        algo.fit(train)
        _log.info("testing %d users", test.user.nunique())
        return batch.predict(algo, test, n_jobs=1)

    preds = pd.concat(
        (eval(train, test) for (train, test) in xf.partition_users(ratings, 5, xf.SampleFrac(0.2)))
    )
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.70, abs=0.025)

    user_rmse = preds.groupby("user").apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.90, abs=0.05)


@lktu.wantjit
@mark.slow
def test_ii_known_preds():
    from lenskit import batch

    algo = knn.ItemItem(20, min_sim=1.0e-6)
    _log.info("training %s on ml data", algo)
    algo.fit(lktu.ml_test.ratings)
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
@mark.skipif(not lktu.ml100k.available, reason="ML100K not available")
@mark.parametrize("ncpus", [1, 2])
def test_ii_batch_recommend(ncpus):
    import lenskit.crossfold as xf
    from lenskit import topn

    ratings = lktu.ml100k.ratings

    def eval(train, test):
        _log.info("running training")
        algo = knn.ItemItem(30)
        algo = Recommender.adapt(algo)
        algo.fit(train)
        _log.info("testing %d users", test.user.nunique())
        recs = batch.recommend(algo, test.user.unique(), 100, n_jobs=ncpus)
        return recs

    test_frames = []
    recs = []
    for train, test in xf.partition_users(ratings, 5, xf.SampleFrac(0.2)):
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


def _build_predict(ratings, fold):
    algo = Fallback(knn.ItemItem(20), Bias(damping=5))
    train = ratings[ratings["partition"] != fold]
    algo.fit(train)

    test = ratings[ratings["partition"] == fold]
    preds = batch.predict(algo, test, n_jobs=1)
    return preds
