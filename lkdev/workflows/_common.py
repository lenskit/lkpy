from typing import Any

from lkdev.ghactions import GHStep

PACKAGES = ["lenskit", "lenskit-funksvd", "lenskit-implicit"]


def step_checkout(options: Any = None) -> GHStep:
    return {
        "name": "🛒 Checkout",
        "uses": "actions/checkout@v4",
        "with": {"fetch-depth": 0},
    }
