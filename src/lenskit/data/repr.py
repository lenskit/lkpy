# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Utility functions for implementing ``__str__`` and ``__repr__`` methods
with consistent syntax.
"""

# pyright: strict
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from types import TracebackType
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HasObjectRepr(Protocol):
    def _lk_object_repr(self) -> ObjectRepr: ...


class ReprWriter:
    out: StringIO
    current_indent: int = 0
    newline: bool = True

    def __init__(self):
        self.out = StringIO()

    def write(self, text: str):
        self._write_text(text, eol=False)

    def writeln(self, text: str = ""):
        self._write_text(text, eol=True)

    def string(self) -> str:
        return self.out.getvalue()

    def _write_text(self, text: str, *, eol: bool = True):
        if eol:
            text = text + "\n"
        lines = text.splitlines(True)
        for line in lines:
            self._write(line)

    def _write(self, text: str):
        if self.newline and self.current_indent:
            self.out.write(" " * self.current_indent)
        self.out.write(text)
        if text:
            self.newline = text[-1] == "\n"

    def indent(self, *, size: int = 2):
        return ReprIndenter(self, size)


class ReprIndenter:
    writer: ReprWriter
    size: int

    def __init__(self, writer: ReprWriter, size: int):
        self.writer = writer
        self.size = size

    def __enter__(self):
        self.writer.current_indent += self.size
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: TracebackType):
        self.writer.current_indent -= self.size


@dataclass
class ObjectRepr:
    tag: str
    words: Sequence[str] = tuple()
    comment: str | None = None
    attrs: Mapping[str, Any] | None = None
    _string: None = None

    def write(self, writer: ReprWriter):
        writer.write(f"<{self.tag}")

        if self.words:
            writer.write(" ")
            writer.write(" ".join(self.words))
        if self.comment:
            writer.write(f" ({self.comment})")
        if self.attrs:
            writer.write(" {\n")
            with writer.indent():
                for k, v in self.attrs.items():
                    if isinstance(v, HasObjectRepr):
                        vor = v._lk_object_repr()
                        writer.write(f"{k}: ")
                        vor.write(writer)
                        writer.writeln()
                    else:
                        writer.writeln(f"{k}: {repr(v)}")
            writer.write("}")

        writer.write(">")

    def string(self) -> str:
        writer = ReprWriter()
        self.write(writer)
        return writer.string()


def object_repr(tag: str, *words: str, comment: str | None = None, **attrs: Any) -> ObjectRepr:
    """
    Construct string "object" representations.
    """

    return ObjectRepr(tag, words, comment, attrs)
