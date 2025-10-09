# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import inspect
import os
import pickle
from time import perf_counter
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from pytest import approx, fixture, mark, skip

from lenskit import batch
from lenskit.data import Dataset, ItemList, RecQuery, from_interactions_df
from lenskit.data.builder import DatasetBuilder
from lenskit.metrics import quick_measure_model
from lenskit.pipeline import Component, topn_pipeline
from lenskit.splitting import split_temporal_fraction
from lenskit.training import Trainable, TrainingOptions

retrain = os.environ.get("LK_TEST_RETRAIN")


class BasicComponentTests:
    component: type[Component]
    configs = []

    def test_instantiate_default(self):
        inst = self.component()
        assert inst is not None

        if self.component.config_class() is not None:
            assert inst.config is not None
        else:
            assert inst.config is None

    def test_default_config_vars(self):
        inst = self.component()
        cfg = inst.dump_config()
        for name, value in cfg.items():
            assert hasattr(inst.config, name)

    def test_default_config_round_trip(self):
        inst = self.component()
        cfg = inst.dump_config()

        i2 = self.component(self.component.validate_config(cfg))
        assert i2 is not inst
        assert isinstance(i2, self.component)
        print(cfg)
        print(i2.dump_config())
        assert i2.dump_config() == cfg

    def test_config_round_trip(self):
        if not self.configs:
            skip("no test configs specified")

        for cfg in self.configs:
            inst = self.component(self.component.validate_config(cfg))
            c1 = inst.dump_config()

            i2 = self.component(self.component.validate_config(c1))
            c2 = i2.dump_config()
            # config may be changed from source (due to normalization), but should
            # round-trip.
            assert c2 == c1


class TrainingTests:
    """
    Common tests for component training.
    """

    component: type[Component]
    config: Any = None

    @fixture(scope="function" if retrain else "class")
    def trained_model(self, ml_ds: Dataset):
        model = self.component(self.config)
        if isinstance(model, Trainable):
            model.train(ml_ds, TrainingOptions())
        yield model

    def test_skip_retrain(self, ml_ds: Dataset):
        model = self.component(self.config)
        if not isinstance(model, Trainable):
            skip(f"component {model.__class__.__name__} is not trainable")

        model.train(ml_ds, TrainingOptions())
        v1_data = pickle.dumps(model)

        # train again
        t1 = perf_counter()
        model.train(ml_ds, TrainingOptions(retrain=False))
        t2 = perf_counter()
        # that should be very fast, let's say 50ms
        assert t2 - t1 < 0.05
        # the model shouldn't have changed
        v2_data = pickle.dumps(model)
        assert v2_data == v1_data


class ScorerTests(TrainingTests):
    """
    Common tests for scorer components.  Many of these just test that the component
    runs, not that it produces correct output.
    """

    component: ClassVar[type[Component]]
    can_score: ClassVar[Literal["some", "known", "all"]] = "known"
    "What can this scorer score?"

    expected_rmse: ClassVar[float | tuple[float, float] | object | None] = None
    "Asserts RMSE either less than the provided expected value or between two values as tuple."
    expected_ndcg: ClassVar[float | tuple[float, float] | object | None] = None
    "Asserts nDCG either greater than the provided expected value or between two values as tuple."

    def invoke_scorer(self, inst: Component, **kwargs):
        sig = inspect.signature(inst)
        args = {n: v for (n, v) in kwargs.items() if n in sig.parameters}
        return inst(**args)

    def verify_models_equivalent(self, orig, copy):
        "Verify that two models are equivalent."
        pass

    def test_score_known(self, rng: np.random.Generator, ml_ds: Dataset, trained_model: Component):
        for u in rng.choice(ml_ds.users.ids(), 100):
            item_nums = rng.choice(ml_ds.item_count, 100, replace=False)
            items = ItemList(item_nums=item_nums, vocabulary=ml_ds.items)
            scored = self.invoke_scorer(trained_model, query=u, items=items)
            assert isinstance(scored, ItemList)
            assert np.all(scored.numbers() == item_nums)
            assert np.all(scored.ids() == items.ids())
            scores = scored.scores()
            assert scores is not None
            if self.can_score in ("known", "all"):
                assert np.all(np.isfinite(scores))

    def test_pickle_roundrip(
        self, rng: np.random.Generator, ml_ds: Dataset, trained_model: Component
    ):
        data = pickle.dumps(trained_model, pickle.HIGHEST_PROTOCOL)
        tm2 = pickle.loads(data)
        self.verify_models_equivalent(trained_model, tm2)

        for u in rng.choice(ml_ds.users.ids(), 100):
            item_nums = rng.choice(ml_ds.item_count, 100, replace=False)
            items = ItemList(item_nums=item_nums, vocabulary=ml_ds.items)
            scored = self.invoke_scorer(trained_model, query=u, items=items)
            assert isinstance(scored, ItemList)

            s2 = self.invoke_scorer(tm2, query=u, items=items)
            assert isinstance(s2, ItemList)

            assert np.all(scored.numbers() == item_nums)
            assert np.all(scored.ids() == items.ids())

            arr = scored.scores()
            assert arr is not None
            arr2 = s2.scores()
            assert arr2 is not None
            assert arr == approx(arr2, nan_ok=True)

    def test_score_unknown_user(
        self, rng: np.random.Generator, ml_ds: Dataset, trained_model: Component
    ):
        "score with an unknown user ID"
        item_nums = rng.choice(ml_ds.item_count, 100, replace=False)
        items = ItemList(item_nums=item_nums, vocabulary=ml_ds.items)

        scored = self.invoke_scorer(trained_model, query=-1348, items=items)
        scores = scored.scores("pandas", index="ids")
        assert scores is not None
        if self.can_score == "all":
            assert np.all(np.isfinite(scores))

    def test_score_unknown_item(
        self, rng: np.random.Generator, ml_ds: Dataset, trained_model: Component
    ):
        "score with one target item unknown"
        item_nums = rng.choice(ml_ds.item_count, 100, replace=False)
        item_ids = ml_ds.items.ids(item_nums).tolist()
        item_ids.append(-318)
        items = ItemList(item_ids)

        scored = self.invoke_scorer(trained_model, query=ml_ds.users.id(0), items=items)
        scores = scored.scores("pandas", index="ids")
        assert scores is not None
        if self.can_score == "all":
            assert np.all(np.isfinite(scores))
        elif self.can_score == "known":
            assert np.all(np.isfinite(scores[:-1]))

    def test_score_empty_query(
        self, rng: np.random.Generator, ml_ds: Dataset, trained_model: Component
    ):
        "score with an empty query"
        item_nums = rng.choice(ml_ds.item_count, 100, replace=False)
        items = ItemList(item_nums=item_nums, vocabulary=ml_ds.items)
        q = RecQuery()
        scored = self.invoke_scorer(trained_model, query=q, items=items)
        assert np.all(scored.numbers() == item_nums)
        assert np.all(scored.ids() == items.ids())
        scores = scored.scores()
        assert scores is not None

    def test_score_query_history(
        self, rng: np.random.Generator, ml_ds: Dataset, trained_model: Component
    ):
        "score when query has user ID and history"
        u = rng.choice(ml_ds.users.ids())
        u_row = ml_ds.user_row(u)

        item_nums = rng.choice(ml_ds.item_count, 100, replace=False)
        items = ItemList(item_nums=item_nums, vocabulary=ml_ds.items)
        q = RecQuery(user_id=u, user_items=u_row)
        scored = self.invoke_scorer(trained_model, query=q, items=items)
        assert np.all(scored.numbers() == item_nums)
        assert np.all(scored.ids() == items.ids())
        scores = scored.scores()
        assert scores is not None

    def test_score_query_history_only(
        self, rng: np.random.Generator, ml_ds: Dataset, trained_model: Component
    ):
        "score when query only has history"
        u = rng.choice(ml_ds.users.ids())
        u_row = ml_ds.user_row(u)

        item_nums = rng.choice(ml_ds.item_count, 100, replace=False)
        items = ItemList(item_nums=item_nums, vocabulary=ml_ds.items)
        q = RecQuery(user_items=u_row)
        scored = self.invoke_scorer(trained_model, query=q, items=items)
        assert np.all(scored.numbers() == item_nums)
        assert np.all(scored.ids() == items.ids())
        scores = scored.scores()
        assert scores is not None

    def test_score_empty_items(
        self, rng: np.random.Generator, ml_ds: Dataset, trained_model: Component
    ):
        "score an empty list of items"
        u = rng.choice(ml_ds.users.ids())

        items = ItemList()
        scored = self.invoke_scorer(trained_model, query=u, items=items)
        assert len(scored) == 0
        scores = scored.scores()
        assert scores is not None

    def test_train_score_items_missing_data(self, rng: np.random.Generator, ml_ds: Dataset):
        "train and score when some entities are missing data"
        drop_i = rng.choice(ml_ds.items.ids(), 20)
        drop_u = rng.choice(ml_ds.users.ids(), 5)

        dsb = DatasetBuilder(ml_ds)
        df = ml_ds.interactions().pandas(ids=True)
        df = df[~(df["user_id"].isin(drop_u))]
        df = df[~(df["item_id"].isin(drop_i))]

        iname = ml_ds.default_interaction_class()
        dsb.clear_relationships(iname)
        dsb.add_relationships(iname, df, entities=["user", "item"])
        ds = dsb.build()

        model = self.component(self.config)
        assert isinstance(model, Trainable)
        model.train(ds, TrainingOptions())

        good_u = rng.choice(ml_ds.users.ids(), 10, replace=False)
        for u in set(good_u) | set(drop_u):
            items = rng.choice(ml_ds.items.ids(), 50, replace=False)
            items = np.unique(np.concatenate([items, rng.choice(drop_i, 5)]))
            items = ItemList(items, vocabulary=ds.items)

            scored = self.invoke_scorer(model, query=u, items=items)
            assert len(scored) == len(items)
            assert np.all(scored.numbers() == items.numbers())
            assert np.all(scored.ids() == items.ids())

            scores = scored.scores()
            assert scores is not None

    @mark.slow
    def test_train_recommend(self, ml_ds: Dataset):
        """
        Test that a full train-recommend pipeline works.
        """
        split = split_temporal_fraction(ml_ds, 0.2)
        model = self.component(self.config)
        pipe = topn_pipeline(model)
        pipe.train(split.train)

        recs = batch.recommend(pipe, split.test)
        assert len(recs) == len(split.test)

    @mark.slow
    def test_run_with_doubles(self, ml_ratings: pd.DataFrame):
        ml_ratings = ml_ratings.astype({"rating": "f8"})
        ml_ds = from_interactions_df(ml_ratings)
        split = split_temporal_fraction(ml_ds, 0.3)
        model = self.component(self.config)
        pipe = topn_pipeline(model)
        pipe.train(split.train)

        recs = batch.recommend(pipe, split.test)
        assert len(recs) == len(split.test)

    @mark.slow
    @mark.eval
    def test_batch_prediction_accuracy(self, rng: np.random.Generator, ml_100k: pd.DataFrame):
        if self.expected_rmse is None:
            skip("expected RMSE not defined")

        ml_ds = from_interactions_df(ml_100k)
        model = self.component(self.config)
        eval_result = quick_measure_model(model, ml_ds, predicts_ratings=True, rng=rng)
        rmse = float(eval_result.list_summary().loc["RMSE", "mean"])  # type: ignore
        if isinstance(self.expected_rmse, tuple):
            assert self.expected_rmse[0] <= rmse <= self.expected_rmse[1]
        elif isinstance(self.expected_rmse, float):
            assert rmse < self.expected_rmse
        else:
            assert rmse == self.expected_rmse

    @mark.slow
    @mark.eval
    def test_batch_top_n_accuracy(self, rng: np.random.Generator, ml_100k: pd.DataFrame):
        if self.expected_ndcg is None:
            skip("expected nDCG not defined")

        ml_ds = from_interactions_df(ml_100k)
        model = self.component(self.config)
        eval_result = quick_measure_model(model, ml_ds, predicts_ratings=True, rng=rng)
        ndcg = float(eval_result.list_summary().loc["NDCG", "mean"])  # type: ignore
        if isinstance(self.expected_ndcg, tuple):
            assert self.expected_ndcg[0] <= ndcg <= self.expected_ndcg[1]
        elif isinstance(self.expected_ndcg, float):
            assert ndcg >= self.expected_ndcg
        else:
            assert ndcg == self.expected_ndcg
