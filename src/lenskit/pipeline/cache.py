"""
Pipeline caching support.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import JsonValue, TypeAdapter

from lenskit.logging import get_logger

from .components import ComponentConstructor
from .types import type_string

_log = get_logger(__name__)


@dataclass(frozen=True)
class PipelineCacheKey:
    ctor_name: str
    ctor: Callable[..., Any] = field(hash=False)
    config: Mapping[str, JsonValue] | None


class PipelineCache:
    """
    A cache to share components between pipelines.

    This cache can be used by :class:`~lenskit.pipeline.PipelineBuilder` to
    cache pipeline component instances between multiple pipelines using the same
    components with the same configuration.
    """

    _cache: dict[PipelineCacheKey, Any]

    def __init__(self):
        self._cache = {}

    def _make_key(self, ctor: ComponentConstructor[Any], config: object | None):
        name = type_string(ctor)  # type: ignore
        if config is not None:
            config = TypeAdapter[Any](ctor.config_class()).dump_python(config, mode="json")
            assert isinstance(config, dict)

        return PipelineCacheKey(name, ctor, config)

    def get_cached(self, ctor: ComponentConstructor[Any], config: object | None):
        key = self._make_key(ctor, object)
        return self._cache.get(key, None)

    def cache(self, ctor: ComponentConstructor[Any], config: object | None, instance: object):
        key = self._make_key(ctor, object)
        self._cache[key] = instance

    def get_instance(self, ctor: ComponentConstructor[Any], config: object | None):
        """
        Get a component instance from the cache, creating if it necessry.
        """
        key = self._make_key(ctor, config)
        instance = self._cache.get(key, None)
        if instance is None:
            instance = ctor(config)
            self._cache[key] = instance
            _log.debug(
                "instantiated component", component=ctor, config=config, n_cached=len(self._cache)
            )
        else:
            _log.debug(
                "found cached component", component=ctor, config=config, n_cached=len(self._cache)
            )
        return instance
