from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from pytest import importorskip

from lenskit.schemas.tuning import SearchSpace

if TYPE_CHECKING:
    from lenskit.tuning._optuna import point
else:
    point = importorskip("lenskit.tuning._optuna.point")


def test_nested_space():
    space = TypeAdapter(SearchSpace).validate_python(
        {
            "bias": {
                "user": {"type": "float", "min": 1.0e-9, "max": 100, "scale": "log"},
                "item": {"type": "float", "min": 1.0e-9, "max": 100, "scale": "log"},
            }
        }
    )
    pt = point.SearchPoint(space, {"bias.user": 1.0, "bias.item": 3.0})
    cfg = pt.to_config()
    assert cfg["bias"]["user"] == 1.0
    assert cfg["bias"]["item"] == 3.0


def test_pow2_space():
    space = TypeAdapter(SearchSpace).validate_python(
        {"embedding_size": {"type": "int", "min": 8, "max": 1024, "scale": "pow2"}}
    )
    pt = point.SearchPoint(space, {"embedding_size%pow2": 5})
    cfg = pt.to_config()
    assert cfg == {"embedding_size": 32}
