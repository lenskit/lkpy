from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from pytest import importorskip

from lenskit.schemas.tuning import SearchSpace

if TYPE_CHECKING:
    from lenskit.tuning._optuna import point
else:
    point = importorskip("lenskit.tuning._optuna.point")


def test_nested_space():
    pt = point.SearchPoint({"bias.user": 1.0, "bias.item": 3.0})
    cfg = pt.to_config()
    assert cfg["bias"]["user"] == 1.0
    assert cfg["bias"]["item"] == 3.0


def test_pow2_space():
    pt = point.SearchPoint({"embedding_size%pow2": 5})
    cfg = pt.to_config()
    assert cfg == {"embedding_size": 32}
