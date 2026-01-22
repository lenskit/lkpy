# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from contextvars import ContextVar
from typing import Any

from lenskit.logging import get_logger
from lenskit.pipeline import PipelineBuilder
from lenskit.pipeline.nodes import ComponentInstanceNode
from lenskit.pipeline.types import Lazy, T

_log = get_logger(__name__)
hook_calls = ContextVar(f"lk-{__name__}-hook-calls", default=[])


def proc_hello(msg: str) -> str:
    return f"Hello, {msg}"


def proc_prefix(msg: str, prefix: str) -> str:
    return prefix + msg


def lazy_prefix(msg: Lazy[str], prefix: str, extra: Lazy[str]) -> str:
    return prefix + msg.get()


def _input_hook(
    node: ComponentInstanceNode[Any], input_name: str, input_type: Any, value: Any, **context
) -> Any:
    cs = hook_calls.get()
    _log.debug("input hook called", n=len(cs), msg=value)
    cs.append((node.name, input_name))
    return value


def test_component_input_called():
    build = PipelineBuilder()
    msg = build.create_input("message", str)
    hello = build.add_component("hello", proc_hello, msg=msg)

    build.add_run_hook("component-input", _input_hook)

    pipe = build.build()

    assert len(pipe.config.hooks.run["component-input"]) == 1
    assert pipe.config.hooks.run["component-input"][0].function.endswith(
        "tests.pipeline.test_component_input_hook:_input_hook"
    )
    assert pipe.config.hooks.run["component-input"][0].priority == 1

    hook_calls.set([])
    out = pipe.run(hello, message="world")
    assert out == "Hello, world"

    assert len(hook_calls.get()) == 1


def test_hook_passed_to_clone():
    build = PipelineBuilder()
    msg = build.create_input("message", str)
    _hello = build.add_component("hello", proc_hello, msg=msg)

    build.add_run_hook("component-input", _input_hook)

    pipe = build.build()

    b2 = PipelineBuilder.from_pipeline(pipe)
    cfg2 = b2.build_config()

    assert len(cfg2.hooks.run["component-input"]) == 1
    assert cfg2.hooks.run["component-input"][0].function.endswith(
        "tests.pipeline.test_component_input_hook:_input_hook"
    )
    assert cfg2.hooks.run["component-input"][0].priority == 1


def test_hook_loaded_from_config():
    build = PipelineBuilder()
    msg = build.create_input("message", str)
    _hello = build.add_component("hello", proc_hello, msg=msg)

    build.add_run_hook("component-input", _input_hook)

    cfg = build.build_config()
    print(cfg.model_dump_json(indent=2))

    b2 = PipelineBuilder.from_config(cfg)
    cfg2 = b2.build_config()
    print(cfg2.model_dump_json(indent=2))

    assert len(cfg2.hooks.run["component-input"]) == 1
    assert cfg2.hooks.run["component-input"][0].function.endswith(
        "tests.pipeline.test_component_input_hook:_input_hook"
    )
    assert cfg2.hooks.run["component-input"][0].priority == 1


def test_component_input_called_multi():
    build = PipelineBuilder()
    msg = build.create_input("message", str)
    pfx = build.add_component("prefix", proc_prefix, msg=msg, prefix="good ")
    hello = build.add_component("hello", proc_hello, msg=pfx)

    build.add_run_hook("component-input", _input_hook)

    pipe = build.build()

    hook_calls.set([])
    out = pipe.run(hello, message="friend")
    assert out == "Hello, good friend"

    # we have 3 input edges: 2 to proc_prefix, 1 to proc_hello
    cs = hook_calls.get()
    assert len(hook_calls.get()) == 3
    assert cs == [("prefix", "msg"), ("prefix", "prefix"), ("hello", "msg")]


def test_component_lazy_hook():
    build = PipelineBuilder()
    msg = build.create_input("message", str)
    pfx = build.add_component("prefix", lazy_prefix, msg=msg, prefix="good ", extra="NOTHING")

    build.add_run_hook("component-input", _input_hook)

    pipe = build.build()

    hook_calls.set([])
    out = pipe.run(pfx, message="friend")
    assert out == "good friend"

    # we should be called twice: once for msg, once for prefix. extra is not called.
    cs = hook_calls.get()
    assert len(hook_calls.get()) == 2
    assert cs == [
        ("prefix", "prefix"),
        ("prefix", "msg"),
    ]
