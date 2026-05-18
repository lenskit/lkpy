# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from contextlib import contextmanager
from typing import TextIO


class IndentWriter:
    """
    Write output with configurable indents.
    """

    level: int = 4
    width: int
    _target: TextIO

    def __init__(self, target: TextIO, *, width: int = 4):
        self.width = width
        self._target = target

    def print(self, data: str = "", *, eol: bool = True):
        """
        Print another message to the output.
        """
        if data:
            self._target.write(" " * self.level)
            self._target.write(data)
        if eol:
            self._target.write("\n")

    def add_indent(self):
        self.level += self.width

    def drop_indent(self):
        self.level -= self.width
        assert self.level >= 0

    @contextmanager
    def indent(self):
        self.add_indent()
        try:
            yield self
        finally:
            self.drop_indent()
