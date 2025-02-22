# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Sequence

from docutils import nodes
from docutils.parsers.rst.directives.admonitions import Admonition
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

STABILITY_LEVELS = {
    "full": """
This API is at the **full** stability level; breaking changes for both callers
and implementers will be reserved for annual major version bumps. See
:ref:`stability-levels` for details.
""",
    "caller": """
This API is at the **caller** stability level: breaking changes for code calling
this function or class will be reserved for annual major version bumps, but minor
versions may introduce changes that break subclasses or reimplementations. See
:ref:`stability-levels` for details.
""",
    "testing": """
This API is at the **testing**: we will avoid gratuituous breaking changes for
callers, but may make such changes in minor versions with clear statements in the
release notes.  No stability guarantees are made for subclasses or re-implementers.
See :ref:`stability-levels` for details.
""",
    "internal": """
This API is at the **internal** or **experimental** stability level: it may
change at any time, and breaking changes will not necessarily be described in
the release notes. See :ref:`stability-levels` for details.
""",
}


class StabilityDirective(Admonition):
    """
    A directive to note an API's stability.
    """

    required_arguments = 1

    def run(self) -> Sequence[nodes.Node]:
        level = self.arguments[0]
        text = STABILITY_LEVELS.get(level, "")
        self.arguments = ["Stability: " + level.capitalize()]
        self.options["class"] = ["note"]
        self.content.insert(0, text, "unknown")
        return super().run()


def scan_stability_notes(app, domain, objtype, contentnode):
    for node in contentnode.findall(_is_stability_field):
        vals = node.traverse(nodes.field_body)
        if not vals:
            continue

        body: nodes.field_body = vals[0]
        text = body.astext()
        if text.lower() in STABILITY_LEVELS:
            # we found a stability level — add reference
            body.clear()
            body.extend(
                [
                    nodes.Text(text.capitalize()),
                    nodes.Text(" (see "),
                    nodes.inline(
                        "",
                        "",
                        pending_xref(
                            "",
                            nodes.inline(
                                "",
                                "stability-levels",
                                classes=["xref", "std", "std-ref"],
                            ),
                            refdoc="api/operations",
                            refdomain="std",
                            reftype="ref",
                            reftarget="stability-levels",
                            refexplicit=False,
                            refwarn=True,
                        ),
                    ),
                    nodes.Text(")."),
                ]
            )


def _is_stability_field(node):
    if isinstance(node, nodes.field):
        names = node.traverse(nodes.field_name)
        if names:
            return names[0].astext() == "Stability"


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive("stability", StabilityDirective)
    app.connect("object-description-transform", scan_stability_notes)

    return {
        "version": "2025.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
