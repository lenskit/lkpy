# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy import linalg as la

import pytest
from pytest import approx, fixture, mark

from lenskit.basic.history import UserTrainingHistoryLookup
from lenskit.data import ItemList, ItemListCollection, UserIDKey, Vocabulary, from_interactions_df
from lenskit.diagnostics import DataWarning
from lenskit.knn.item import ItemKNNScorer
from lenskit.pipeline import topn_pipeline
from lenskit.pipeline.common import predict_pipeline
from lenskit.testing import BasicComponentTests, ScorerTests

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
    columns=["user_id", "item_id", "rating"],
)
simple_ds = from_interactions_df(simple_ratings)


@fixture
def ml_subset(ml_ratings):
    "Fixture that returns a subset of the MovieLens database."
    icounts = ml_ratings.groupby("item_id").rating.count()
    top = icounts.nlargest(500)
    top_rates = ml_ratings[ml_ratings["item_id"].isin(top.index)]
    _log.info("top 500 items yield %d of %d ratings", len(top_rates), len(ml_ratings))
    return top_rates


class TestItemKNN(BasicComponentTests, ScorerTests):
    can_score = "some"
    component = ItemKNNScorer
    expected_rmse = (0.85, 0.95)
    expected_ndcg = 0.03

    def verify_models_equivalent(self, orig: ItemKNNScorer, copy: ItemKNNScorer):
        o_mat = orig.sim_matrix.to_scipy()
        r_mat = copy.sim_matrix.to_scipy()
        assert all(np.logical_not(np.isnan(r_mat.data)))
        assert all(r_mat.data > 0)
        # a little tolerance
        assert all(r_mat.data < 1 + 1.0e-6)

        assert all(copy.item_counts == orig.item_counts)
        assert copy.item_counts.sum() == len(r_mat.data)
        assert len(r_mat.data) == len(r_mat.data)
        assert all(r_mat.indptr == o_mat.indptr)
        assert r_mat.data == approx(o_mat.data)

        assert all(r_mat.indptr == o_mat.indptr)


def test_ii_config():
    model = ItemKNNScorer(k=30)
    cfg = model.dump_config()
    print(cfg)
    assert cfg["feedback"] == "explicit"
    assert cfg["max_nbrs"] == 30


def test_ii_train():
    algo = ItemKNNScorer(k=30, save_nbrs=500)
    algo.train(simple_ds)

    assert isinstance(algo.item_means, np.ndarray)
    assert isinstance(algo.item_counts, np.ndarray)
    matrix = algo.sim_matrix.to_scipy()

    test_means = simple_ratings.groupby("item_id")["rating"].mean()
    test_means = test_means.reindex(algo.items.ids())
    assert np.all(algo.item_means == test_means.values.astype("f4"))

    # 6 is a neighbor of 7
    six, seven = algo.items.numbers([6, 7])
    _log.info("six: %d", six)
    _log.info("seven: %d", seven)
    _log.info("matrix: %s", algo.sim_matrix)
    assert matrix[six, seven] > 0
    # and has the correct score
    six_v = simple_ratings[simple_ratings["item_id"] == 6].set_index("user_id").rating
    six_v = six_v - six_v.mean()
    seven_v = simple_ratings[simple_ratings["item_id"] == 7].set_index("user_id").rating
    seven_v = seven_v - seven_v.mean()
    denom = la.norm(six_v.values) * la.norm(seven_v.values)
    six_v, seven_v = six_v.align(seven_v, join="inner")
    num = six_v.dot(seven_v)
    assert matrix[six, seven] == approx(num / denom, 0.01)  # type: ignore

    assert all(np.logical_not(np.isnan(matrix.data)))
    assert all(matrix.data > 0)
    # a little tolerance
    assert all(matrix.data < 1 + 1.0e-6)


def test_ii_train_unbounded():
    algo = ItemKNNScorer(k=30)
    algo.train(simple_ds)

    matrix = algo.sim_matrix.to_scipy()
    assert all(np.logical_not(np.isnan(matrix.data)))
    assert all(matrix.data > 0)
    # a little tolerance
    assert all(matrix.data < 1 + 1.0e-6)

    # 6 is a neighbor of 7
    six, seven = algo.items.numbers([6, 7])
    assert matrix[six, seven] > 0

    # and has the correct score
    six_v = simple_ratings[simple_ratings["item_id"] == 6].set_index("user_id").rating
    six_v = six_v - six_v.mean()
    seven_v = simple_ratings[simple_ratings["item_id"] == 7].set_index("user_id").rating
    seven_v = seven_v - seven_v.mean()
    denom = la.norm(six_v.values) * la.norm(seven_v.values)
    six_v, seven_v = six_v.align(seven_v, join="inner")
    num = six_v.dot(seven_v)
    assert matrix[six, seven] == approx(num / denom, 0.01)  # type: ignore


def test_ii_simple_predict():
    history = UserTrainingHistoryLookup()
    history.train(simple_ds)
    algo = ItemKNNScorer(k=30, save_nbrs=500)
    algo.train(simple_ds)

    q = history(3)
    res = algo(q, ItemList([6]))
    _log.info("got predictions: %s", res)
    assert res is not None
    assert len(res) == 1
    assert 6 in res.ids()
    assert not np.isnan(res.scores()[0])


def test_ii_simple_implicit_predict():
    history = UserTrainingHistoryLookup()
    history.train(simple_ds)
    algo = ItemKNNScorer(k=30, feedback="implicit")
    algo.train(from_interactions_df(simple_ratings.loc[:, ["user_id", "item_id"]]))

    q = history(3)
    res = algo(q, ItemList([6]))
    assert res is not None
    assert len(res) == 1
    assert 6 in res.ids()
    assert not np.isnan(res.scores()[0])
    assert np.all(res.scores() > 0)


def test_ii_simple_predict_unknown():
    history = UserTrainingHistoryLookup()
    history.train(simple_ds)
    algo = ItemKNNScorer(k=30, save_nbrs=500)
    algo.train(simple_ds)

    q = history(3)
    res = algo(q, ItemList([6, 100]))
    _log.info("got predictions: %s", res)
    assert res is not None
    assert len(res) == 2
    assert res.ids().tolist() == [6, 100]
    assert not np.isnan(res.scores()[0])
    assert np.isnan(res.scores()[1])


def test_ii_warns_center():
    "Test that item-item warns if you center non-centerable data"
    data = simple_ratings.assign(rating=1)
    algo = ItemKNNScorer(k=5)
    with pytest.warns(DataWarning):
        algo.train(from_interactions_df(data))


@mark.slow
def test_ii_train_ml100k(tmp_path, ml_100k):
    "Test an unbounded model on ML-100K"
    algo = ItemKNNScorer(k=30)
    _log.info("training model")
    algo.train(from_interactions_df(ml_100k))

    _log.info("testing model")

    matrix = algo.sim_matrix.to_scipy()
    assert all(np.logical_not(np.isnan(matrix.data)))
    assert np.all(matrix.data > 0)

    # a little tolerance
    assert np.max(matrix.data) <= 1

    assert algo.item_counts.sum() == len(matrix.data)

    means = ml_100k.groupby("item_id").rating.mean()
    assert means[algo.items.ids()].values == approx(algo.item_means)

    # save
    fn = tmp_path / "ii.mod"
    _log.info("saving model to %s", fn)
    with fn.open("wb") as modf:
        pickle.dump(algo, modf)

    _log.info("reloading model")
    with fn.open("rb") as modf:
        restored = pickle.load(modf)

    r_mat = restored.sim_matrix.to_scipy()

    assert all(r_mat.data > 0)
    assert all(r_mat.data == matrix.data)


@mark.slow
def test_ii_large_models(rng, ml_ratings, ml_ds):
    "Several tests of large trained I-I models"
    _log.info("training limited model")
    MODEL_SIZE = 100
    algo_lim = ItemKNNScorer(k=30, save_nbrs=MODEL_SIZE)
    algo_lim.train(ml_ds)

    _log.info("training unbounded model")
    algo_ub = ItemKNNScorer(k=30)
    algo_ub.train(ml_ds)

    _log.info("testing models")
    mat_lim = algo_lim.sim_matrix.to_scipy()
    assert all(np.logical_not(np.isnan(mat_lim.data)))
    assert mat_lim.data.min() > 0
    # a little tolerance
    assert mat_lim.data.max() <= 1 + 1.0e-6

    means = ml_ratings.groupby("item_id").rating.mean()
    means = means.reindex(algo_ub.items.ids(), fill_value=0.0)
    assert means.values == approx(algo_lim.item_means)

    mat_ub = algo_ub.sim_matrix.to_scipy()
    assert all(np.logical_not(np.isnan(mat_ub.data)))
    assert mat_ub.data.min() > 0
    assert mat_ub.data.max() <= 1 + 1.0e-6

    means = ml_ratings.groupby("item_id").rating.mean()
    means = means.reindex(algo_ub.items.ids(), fill_value=0.0)
    assert means.values == approx(algo_ub.item_means)

    mc_rates = (
        ml_ratings.set_index("item_id")
        .join(pd.DataFrame({"item_mean": means}))
        .assign(rating=lambda df: df.rating - df.item_mean)
    )

    # _log.info("make sure the similarity matrix is sorted")
    # for i in range(algo_lim.items.size):
    #     sp = mat_lim.indptr[i]
    #     ep = mat_lim.indptr[i + 1]
    #     vals = mat_lim.data[sp:ep]
    #     diffs = np.diff(vals)
    #     if np.any(diffs > 0):
    #         _log.error("row %d: %d non-sorted values", i, np.sum(diffs <= 0))
    #         (bad,) = np.nonzero(diffs > 0)
    #         for i in bad:
    #             _log.info("bad indices %d: %d %d", i, vals[i], vals[i + 1])
    #         raise AssertionError(f"{np.sum(diffs <= 0)} non-sorted values")

    _log.info("checking a sample of neighborhoods")
    items = algo_ub.items.ids()
    items = items[algo_ub.item_counts > 0]
    for i in rng.choice(items, 50):
        ipos = algo_ub.items.number(i)
        _log.debug("checking item %d at position %d", i, ipos)
        assert ipos == algo_lim.items.number(i)
        irates = mc_rates.loc[[i], :].set_index("user_id").rating

        ub_row = mat_ub[[ipos]]
        b_row = mat_lim[[ipos]]
        assert len(b_row.data) <= MODEL_SIZE
        ub_cols = ub_row.indices
        b_cols = b_row.indices
        _log.debug("kept %d of %d neighbors", len(b_cols), len(ub_cols))

        # all bounded columns are in the unbounded columns
        _log.debug("checking that bounded columns are a subset of unbounded")
        present = np.isin(b_cols, ub_cols)
        if not np.all(present):
            _log.error("missing items: %s", b_cols[~present])
            _log.error("scores: %s", b_row.data[~present])  # type: ignore
            raise AssertionError(f"missing {np.sum(~present)} values from unbounded")

        # spot-check some similarities
        _log.debug("checking equal similarities")
        for n in rng.choice(ub_cols, min(10, len(ub_cols))):
            n_id = algo_ub.items.id(n)
            n_rates = mc_rates.loc[n_id, :].set_index("user_id").rating
            ir, nr = irates.align(n_rates, fill_value=0)
            cor = ir.corr(nr)
            assert mat_ub[ipos, n] == approx(cor, abs=1.0e-6)

        # short rows are equal
        if len(b_cols) < MODEL_SIZE:
            _log.debug("short row of length %d", len(b_cols))
            assert len(b_row) == len(ub_row)
            assert b_row.data == approx(ub_row.data)
            continue

        # row is truncated - check that truncation is correct
        ub_nbrs = pd.Series(ub_row.data, algo_ub.items.ids(ub_cols))
        b_nbrs = pd.Series(b_row.data, algo_lim.items.ids(b_cols))

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


@mark.slow
def test_ii_implicit_large(rng, ml_ratings):
    "Test that implicit-feedback mode works on full test data."
    _log.info("training model")
    NBRS = 5
    NUSERS = 25
    NRECS = 50
    algo = ItemKNNScorer(k=NBRS, feedback="implicit")
    pipe = topn_pipeline(algo)
    pipe.train(from_interactions_df(ml_ratings[["user_id", "item_id"]], item_col="item_id"))

    users = rng.choice(ml_ratings["user_id"].unique(), NUSERS)

    items: Vocabulary = algo.items
    mat: NDArray[np.float32] = algo.sim_matrix.to_scipy().toarray()

    for user in users:
        recs = pipe.run("recommender", query=user, n=NRECS)
        _log.info("user %s recs\n%r", user, recs)
        assert isinstance(recs, ItemList)
        assert len(recs) == NRECS
        urates = ml_ratings[ml_ratings["user_id"] == user]

        smat = mat[items.numbers(urates["item_id"].values), :]
        for row in recs.to_df().itertuples():
            col = smat[:, items.number(row.item_id)]
            top, _is = torch.topk(torch.from_numpy(col), NBRS)
            score = top.sum()
            try:
                assert row.score == approx(score)
            except AssertionError as e:
                _log.error("test failed for user %s item %s", user, row.item_id)
                _log.info("score: %.6f", row.score)
                _log.info("sims:\n%s", col)
                _log.info("total: %.3f", col.sum())
                _log.info("filtered: %s", top)
                _log.info("filtered sum: %.3f", top.sum())
                raise e


def test_ii_known_preds(ml_ds):
    from lenskit import batch

    iknn = ItemKNNScorer(k=20, min_sim=1.0e-6)
    pipe = predict_pipeline(iknn, fallback=False)  # noqa: F821
    _log.info("training %s on ml data", iknn)
    pipe.train(ml_ds)
    _log.info("model means: %s", iknn.item_means)

    dir = Path(__file__).parent
    pred_file = dir / "item-item-preds.csv"
    _log.info("reading known predictions from %s", pred_file)
    known_preds = pd.read_csv(str(pred_file))
    known = ItemListCollection.from_df(known_preds, UserIDKey)

    preds = batch.predict(pipe, known, n_jobs=1)
    preds = preds.to_df().drop(columns=["prediction"], errors="ignore")

    merged = pd.merge(known_preds.rename(columns={"prediction": "expected"}), preds)
    assert len(merged) == len(preds)
    merged["error"] = merged.expected - merged.score
    try:
        assert not any(merged.score.isna() & merged.expected.notna())
    except AssertionError as e:
        bad = merged[merged.score.isna() & merged.expected.notna()]
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
