# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Render pipelines to diagrams.
"""

import io
from typing import Literal, TextIO, overload

from lenskit.util import IndentWriter

from ._impl import Pipeline


@overload
def render_pipeline_mmd(
    pipe: Pipeline, *, direction: Literal["LR", "TB"] = "LR", out: None = None
) -> str: ...
@overload
def render_pipeline_mmd(
    pipe: Pipeline, *, direction: Literal["LR", "TB"] = "LR", out: TextIO
) -> None: ...
def render_pipeline_mmd(
    pipe: Pipeline, *, direction: Literal["LR", "TB"] = "LR", out: TextIO | None = None
) -> str | None:
    rv = False
    if out is None:
        out = io.StringIO()
        rv = True

    cfg = pipe.config

    w = IndentWriter(out)

    w.print(f"flowchart {direction}")
    with w.indent():
        w.print("classDef optional stroke-dasharray: 5 5;")
        w.print('subgraph input["Inputs"]')
        with w.indent():
            for pi in cfg.inputs:
                w.print(f'{pi.name}[/"{pi.name}"/]')
        w.print("end")

        w.print()
        for name, comp in cfg.components.items():
            w.print(f'{name}["{name}\\n{comp.code}"]')

        w.print()
        for name, comp in cfg.components.items():
            for iname, isrc in comp.inputs.items():
                w.print(f"{isrc} -- {iname} --> {name}")

    if rv:
        assert isinstance(out, io.StringIO)
        return out.getvalue()
