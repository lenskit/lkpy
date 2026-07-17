import logging
import math
from random import choices
from typing import TYPE_CHECKING

from pyarrow.compute import AssumeTimezoneOptions
from pydantic import TypeAdapter

import hypothesis.strategies as st
from hypothesis import assume, given
from pytest import fixture, importorskip

from lenskit.schemas.tuning import SearchParam, SearchSpace

if TYPE_CHECKING:
    import optuna

    from lenskit.tuning._optuna import point
else:
    point = importorskip("lenskit.tuning._optuna.point")
    optuna = importorskip("optuna")

NTRIALS = 10


@fixture(scope="function", autouse=True)
def disable_optuna_log(caplog):
    with caplog.at_level(logging.WARNING, logger="optuna"):
        yield


@st.composite
def int_params(draw: st.DrawFn):
    scale = draw(st.sampled_from(["uniform", "log", "pow2"]))
    if scale == "uniform":
        min = draw(st.integers(min_value=0, max_value=128))
    elif scale == "pow2":
        min = 2 ** draw(st.integers(min_value=0, max_value=3))
    else:
        min = draw(st.integers(min_value=1, max_value=8))

    if scale == "pow2":
        max = draw(st.integers(min_value=min * 2, max_value=16 * 1024))
    else:
        max = draw(st.integers(min_value=min, max_value=16 * 1024))

    return SearchParam(type="int", scale=scale, min=min, max=max)


@st.composite
def float_params(draw: st.DrawFn):
    scale = draw(st.sampled_from(["uniform", "log"]))
    if scale == "log":
        min = draw(
            st.floats(
                0,
                1e12,
                exclude_min=True,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            )
        )
    else:
        min = draw(
            st.floats(
                min_value=-1e12,
                max_value=1e12,
                allow_infinity=False,
                allow_nan=False,
                allow_subnormal=False,
            )
        )

    max = draw(
        st.floats(min, max_value=1e15, allow_infinity=False, allow_nan=False, allow_subnormal=False)
    )
    return SearchParam(type="float", scale=scale, min=min, max=max)


BOOL_PARAM = SearchParam(type="bool")
CHOICE_PARAM = SearchParam(type="choice", choices=["pancakes", "waffles"])


@given(int_params())
def test_int_points(param):
    study = optuna.create_study()
    space = {"n": param}

    for _t in range(NTRIALS):
        trial = study.ask()
        pt = point.SearchPoint.ask(space, trial)
        x = pt.to_config()["n"]
        assert isinstance(x, int)
        assert x >= param.min
        assert x <= param.max
        if param.scale == "pow2":
            base = math.log2(x)
            assert int(base) == base


@given(float_params())
def test_float_points(param):
    study = optuna.create_study()
    space = {"x": param}

    for _t in range(NTRIALS):
        trial = study.ask()
        pt = point.SearchPoint.ask(space, trial)
        x = pt.to_config()["x"]
        assert isinstance(x, float)
        assert x >= param.min
        assert x <= param.max


def test_bool_point():
    study = optuna.create_study()
    space = {"x": BOOL_PARAM}

    for _t in range(NTRIALS):
        trial = study.ask()
        pt = point.SearchPoint.ask(space, trial)
        x = pt.to_config()["x"]
        assert isinstance(x, bool)


def test_choice_point():
    study = optuna.create_study()
    space = {"x": CHOICE_PARAM}

    for _t in range(NTRIALS):
        trial = study.ask()
        pt = point.SearchPoint.ask(space, trial)
        x = pt.to_config()["x"]
        assert isinstance(x, str)
        assert x in CHOICE_PARAM.choices


@given(
    st.dictionaries(
        st.from_regex(r"^\w+$"),
        st.one_of(st.just(BOOL_PARAM), st.just(CHOICE_PARAM), int_params(), float_params()),
        max_size=20,
    )
)
def test_multi_space(space: SearchSpace):
    study = optuna.create_study()

    for _t in range(NTRIALS):
        trial = study.ask()
        pt = point.SearchPoint.ask(space, trial)
        cfg = pt.to_config()
        assert len(pt.params) == len(space)
        for name, sp in space.items():
            assert isinstance(sp, SearchParam)
            if sp.scale == "pow2":
                assert f"{name}%pow2" in pt.params
            else:
                assert name in pt.params
            assert name in cfg


def test_nested_space():
    pt = point.SearchPoint({"bias.user": 1.0, "bias.item": 3.0})
    cfg = pt.to_config()
    assert cfg["bias"]["user"] == 1.0
    assert cfg["bias"]["item"] == 3.0


@given(
    st.dictionaries(
        st.from_regex(r"^[a-z]\w+$"),
        st.one_of(
            st.just(BOOL_PARAM),
            st.just(CHOICE_PARAM),
            int_params(),
            float_params(),
            st.dictionaries(
                st.from_regex(r"^[a-z]\w+$"),
                st.one_of(st.just(BOOL_PARAM), st.just(CHOICE_PARAM), int_params(), float_params()),
                min_size=1,
                max_size=5,
            ),
        ),
        max_size=20,
    )
)
def test_nested_spaces(space: SearchSpace):
    study = optuna.create_study()

    for _t in range(NTRIALS):
        trial = study.ask()
        pt = point.SearchPoint.ask(space, trial)
        cfg = pt.to_config()
        for name, sp in space.items():
            assert name in cfg
            if isinstance(sp, SearchParam):
                if sp.scale == "pow2":
                    assert f"{name}%pow2" in pt.params
                else:
                    assert name in pt.params
                assert not isinstance(cfg[name], dict)
            else:
                assert isinstance(cfg[name], dict)
                for n2, sp2 in sp.items():
                    assert isinstance(sp2, SearchParam)
                    if sp2.scale == "pow2":
                        assert f"{name}.{n2}%pow2" in pt.params
                    else:
                        assert f"{name}.{n2}" in pt.params
                    assert n2 in cfg[name]


def test_pow2_space():
    pt = point.SearchPoint({"embedding_size%pow2": 5})
    cfg = pt.to_config()
    assert cfg == {"embedding_size": 32}
