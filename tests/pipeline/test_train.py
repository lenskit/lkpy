# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import warnings
from dataclasses import dataclass
from typing import Any, final

from typing_extensions import override

from pytest import warns

from lenskit.data import Dataset, Vocabulary
from lenskit.diagnostics import PipelineWarning
from lenskit.pipeline import Component, PipelineBuilder
from lenskit.training import Trainable, TrainingOptions


def test_train(ml_ds: Dataset):
    pipe = PipelineBuilder()
    item = pipe.create_input("item", int)

    tc: Trainable = TComponent()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipe.add_component("test", tc, item=item)
    pipe.default_component("test")
    assert not tc.is_trained()

    pipe = pipe.build()
    pipe.train(ml_ds)

    # return true for an item that exists
    assert pipe.run(item=500)
    # return false for an item that does not
    assert not pipe.run(item=-100)


def test_retrain(ml_ds: Dataset):
    pipe = PipelineBuilder()
    item = pipe.create_input("item", int)

    tc = TComponent()
    pipe.add_component("test", tc, item=item)
    pipe.default_component("test")
    assert not tc.is_trained()
    assert tc.times_trained == 0

    pipe = pipe.build()
    pipe.train(ml_ds)
    assert tc.times_trained == 1
    # train again
    pipe.train(ml_ds)
    assert tc.times_trained == 2

    # return true for an item that exists
    assert pipe.run(item=500)
    # return false for an item that does not
    assert not pipe.run(item=-100)


def test_skip_retrain(ml_ds: Dataset):
    pipe = PipelineBuilder()
    item = pipe.create_input("item", int)

    tc: Trainable = TComponent()
    pipe.add_component("test", tc, item=item)
    pipe.default_component("test")
    assert tc.times_trained == 0

    pipe = pipe.build()
    pipe.train(ml_ds, TrainingOptions(retrain=False))
    assert tc.times_trained == 1
    pipe.train(ml_ds, TrainingOptions(retrain=False))
    assert tc.times_trained == 1

    # return true for an item that exists
    assert pipe.run(item=500)
    # return false for an item that does not
    assert not pipe.run(item=-100)


def test_warn_incomplete_trainable():
    "Check that adding a trainable class without is_trained is a warning."
    pipe = PipelineBuilder()
    with warns(PipelineWarning, match=r"does not fully implement"):
        pipe.add_component("partial", PartialTrainable)


def test_warn_incomplete_trainable_instance():
    "Check that adding a trainable object without is_trained is a warning."
    pipe = PipelineBuilder()
    with warns(PipelineWarning, match=r"does not fully implement"):
        pipe.add_component("partial", PartialTrainable())


@dataclass
class TConfig:
    train_limit: int | None = None


@final
class TComponent(Component, Trainable):
    config: TConfig
    items: Vocabulary
    times_trained: int = 0

    def __call__(self, *, item: int) -> bool:
        return self.items.number(item, "none") is not None

    @override
    def is_trained(self):
        return hasattr(self, "items")

    def train(self, data: Dataset, options: TrainingOptions):
        if self.config.train_limit is not None:
            assert self.times_trained >= self.config.train_limit, "trained too many times"

        # we just memorize the items
        self.items = data.items
        self.times_trained += 1

    def get_params(self) -> dict[str, object]:
        return {"items": self.items}

    def load_params(self, params: dict[str, Any]) -> None:
        self.items = params["items"]


class PartialTrainable(Component):
    config: None

    def __call__(self) -> str:
        return "foo"

    def train(self, data: Dataset, options: TrainingOptions):
        pass
