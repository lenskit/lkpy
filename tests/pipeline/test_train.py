# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Any

from lenskit.data.dataset import Dataset
from lenskit.data.vocab import Vocabulary
from lenskit.pipeline import PipelineBuilder
from lenskit.training import Trainable, TrainingOptions


def test_train(ml_ds: Dataset):
    pipe = PipelineBuilder()
    item = pipe.create_input("item", int)

    tc: Trainable = TestComponent()
    pipe.add_component("test", tc, item=item)
    pipe.default_component("test")

    pipe = pipe.build()
    pipe.train(ml_ds)

    # return true for an item that exists
    assert pipe.run(item=500)
    # return false for an item that does not
    assert not pipe.run(item=-100)


class TestComponent:
    items: Vocabulary

    def __call__(self, *, item: int) -> bool:
        return self.items.number(item, "none") is not None

    def train(self, data: Dataset, options: TrainingOptions):
        # we just memorize the items
        self.items = data.items

    def get_params(self) -> dict[str, object]:
        return {"items": self.items}

    def load_params(self, params: dict[str, Any]) -> None:
        self.items = params["items"]
