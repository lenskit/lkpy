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
from abc import ABC, abstractmethod
from typing import Literal, TextIO

from lenskit.util import IndentWriter

from ._impl import Pipeline


class GraphDiagrammer(ABC):
    """
    Render pipeline diagrams.

    Stability:
        Internal
    """

    output: TextIO

    def __init__(self):
        self.output = io.StringIO()

    def set_output(self, out: TextIO):
        self.output = out

    def text(self):
        if isinstance(self.output, io.StringIO):
            return self.output.getvalue()
        else:
            raise RuntimeError("cannot get text for non-string output")

    @abstractmethod
    def render_pipeline(self, pipe: Pipeline): ...


class MermaidDiagrammer(GraphDiagrammer):
    direction: Literal["LR", "TB"]
    label_edges: bool = True
    for_tty: bool = False

    def __init__(self, direction: Literal["LR", "TB"] = "TB", *, for_tty: bool = False):
        super().__init__()
        self.direction = direction
        if for_tty:
            self.for_tty = True
            self.label_edges = False

    def render_pipeline(self, pipe: Pipeline):
        cfg = pipe.config

        w = IndentWriter(self.output)
        br = "\\n" if self.for_tty else "<br>"

        w.print(f"flowchart {self.direction}")
        with w.indent():
            # w.print("classDef optional stroke-dasharray: 5 5;")
            w.print("subgraph Inputs")
            with w.indent():
                for pi in cfg.inputs:
                    w.print(f'{pi.name}[/"{pi.name}"/]')
            w.print("end")

            w.print()
            for name, comp in cfg.components.items():
                names = name
                is_out = name in ("recommender", "rating-predictor")
                for aname, tgt in cfg.aliases.items():
                    if tgt == name:
                        atxt = aname if self.for_tty else f"*{aname}*"
                        names = f"{names}{br}{atxt}"
                        if aname in ("recommender", "rating-predictor"):
                            is_out = True

                if comp.code == "lenskit.pipeline.components:fallback_on_none":
                    w.print(f'{name}{{"{names}"}}')
                else:
                    impl = comp.code
                    impl = re.sub(r"^lenskit\.(?:basic\.\w+:)?", "", impl)
                    if not self.for_tty:
                        impl = f"<tt>{impl}</tt>"
                    label = f"{names}{br}{impl}"
                    if is_out:
                        w.print(f'{name}(["{label}"])')
                    else:
                        w.print(f'{name}["{label}"]')

            w.print()
            for name, comp in cfg.components.items():
                for iname, isrc in comp.inputs.items():
                    if self.label_edges:
                        w.print(f"{isrc} -- {iname} --> {name}")
                    else:
                        w.print(f"{isrc} --> {name}")
