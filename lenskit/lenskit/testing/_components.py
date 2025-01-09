import inspect
import os
import pickle
from typing import ClassVar, Literal

import numpy as np

from pytest import approx, fixture, skip

from lenskit.data import Dataset, ItemList, MatrixDataset, RecQuery
from lenskit.pipeline import Component, Trainable

from ._markers import jit_enabled

retrain = os.environ.get("LK_TEST_RETRAIN")


class BasicComponentTests:
    component: type[Component]
    configs = []

    def test_instantiate_default(self):
        inst = self.component()
        assert inst is not None

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

    needs_jit: ClassVar[bool] = True
    component: type[Component]

    def maybe_skip_nojit(self):
        if self.needs_jit and not jit_enabled:
            skip("JIT is disabled")

    @fixture(scope="function" if retrain else "class")
    def trained_model(self, ml_ds: Dataset):
        self.maybe_skip_nojit()

        model = self.component()
        if isinstance(model, Trainable):
            model.train(ml_ds)
        yield model

    def test_basic_trained(self, ml_ds: Dataset, trained_model: Component):
        assert isinstance(trained_model, self.component)
        if isinstance(trained_model, Trainable):
            assert trained_model.is_trained


class ScorerTests(TrainingTests):
    """
    Common tests for scorer components.  Many of these just test that the component
    runs, not that it produces correct output.
    """

    component: type[Component]
    can_score: ClassVar[Literal["some", "known", "all"]] = "known"
    "What can this scorer score?"

    def invoke_scorer(self, inst: Component, **kwargs):
        sig = inspect.signature(inst)
        args = {n: v for (n, v) in kwargs.items() if n in sig.parameters}
        return inst(**args)

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
        self.maybe_skip_nojit()
        drop_i = rng.choice(ml_ds.items.ids(), 20)
        drop_u = rng.choice(ml_ds.users.ids(), 5)

        df = ml_ds.interaction_log("pandas", fields="all", original_ids=True)
        df = df[~(df["user_id"].isin(drop_u))]
        df = df[~(df["item_id"].isin(drop_i))]
        ds = MatrixDataset(ml_ds.users, ml_ds.items, df)

        model = self.component()
        assert isinstance(model, Trainable)
        model.train(ds)

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
