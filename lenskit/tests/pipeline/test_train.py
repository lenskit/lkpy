# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Any

from lenskit.data.dataset import Dataset
from lenskit.data.vocab import Vocabulary
from lenskit.pipeline import Pipeline
from lenskit.pipeline.components import Trainable


def test_train(ml_ds: Dataset):
    pipe = Pipeline()
    item = pipe.create_input("item", int)

    tc: Trainable[bool] = TestComponent()
    pipe.add_component("test", tc, item=item)

    pipe.train(ml_ds)

    # return true for an item that exists
    assert pipe.run(item=500)
    # return false for an item that does not
    assert not pipe.run(item=-100)


class TestComponent:
    items: Vocabulary

    def __call__(self, *, item: int) -> bool:
        return self.items.number(item, "none") is not None

    @property
    def is_trained(self):
        return hasattr(self, "items")

    def train(self, data: Dataset):
        # we just memorize the items
        self.items = data.items
        return self

    def get_params(self) -> dict[str, object]:
        return {"items": self.items}

    def load_params(self, params: dict[str, Any]) -> None:
        self.items = params["items"]
