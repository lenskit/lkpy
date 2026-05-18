# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Render pipelines to diagrams.
"""

import io
import re
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
            names = name
            for aname, tgt in cfg.aliases.items():
                if tgt == name:
                    names = f"{names}<br><i>{aname}</i>"

            if comp.code == "lenskit.pipeline.components:fallback_on_none":
                w.print(f'{name}{{"{names}"}}')
            else:
                impl = comp.code
                impl = re.sub(r"^lenskit\.(?:basic\.\w+:)", "", impl)
                w.print(f'{name}["{names}<br><tt>{impl}</tt>"]')

        w.print()
        for name, comp in cfg.components.items():
            for iname, isrc in comp.inputs.items():
                w.print(f"{isrc} -- {iname} --> {name}")

    if rv:
        assert isinstance(out, io.StringIO)
        return out.getvalue()
