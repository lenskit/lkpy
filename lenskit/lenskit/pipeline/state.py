# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from collections.abc import Mapping
from typing import Any, Iterator


class PipelineState(Mapping[str, Any]):
    """
    Full results of running a pipeline.  A pipeline state is a dictionary
    mapping node names to their results; it is implemented as a separate class
    instead of directly using a dictionary to allow data to be looked up by node
    aliases in addition to original node names (and to be read-only).

    Client code will generally not construct this class directly.

    Args:
        state:
            The pipeline state to wrap.  The state object stores a reference to
            this dictionary.
        aliases:
            Dictionary of node aliases.
        default:
            The name of the default node (whose data should be returned by
            :attr:`default` ).
    """

    _state: dict[str, Any]
    _aliases: dict[str, str]
    _default: str | None = None

    def __init__(
        self,
        state: dict[str, Any] | None = None,
        aliases: dict[str, str] | None = None,
        default: str | None = None,
    ) -> None:
        self._state = state if state is not None else {}
        self._aliases = aliases if aliases is not None else {}
        self._default = default
        if default is not None and default not in self:
            raise ValueError("default node is not in state or aliases")

    @property
    def default(self) -> Any:
        """
        Return the data from of the default node (typically the last node run).

        Returns:
            The data associated with the default node.

        Raises:
            ValueError: if there is no specified default node.
        """
        if self._default is not None:
            return self[self._default]
        else:
            raise ValueError("pipeline state has no default value")

    @property
    def default_node(self) -> str | None:
        "Return the name of the default node (typically the last node run)."
        return self._default

    def __len__(self):
        return len(self._state)

    def __contains__(self, key: object) -> bool:
        if key in self._state:
            return True
        if key in self._aliases:
            return self._aliases[key] in self
        else:
            return False

    def __getitem__(self, key: str) -> Any:
        if key in self._state:
            return self._state[key]
        elif key in self._aliases:
            return self[self._aliases[key]]
        else:
            raise KeyError(f"pipeline node <{key}>")

    def __iter__(self) -> Iterator[str]:
        return iter(self._state)

    def __str__(self) -> str:
        return f"<PipelineState with {len(self)} nodes>"

    def __repr__(self) -> str:
        return f"<PipelineState with nodes {set(self._state.keys())}>"
