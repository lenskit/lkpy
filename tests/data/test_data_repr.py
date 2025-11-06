# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from lenskit.data.repr import ReprWriter, object_repr


@dataclass
class MsgBox:
    msg: str

    def _lk_object_repr(self):
        return object_repr("MsgBox", self.msg)


def test_empty_writer():
    w = ReprWriter()
    s = w.string()
    assert s == ""


def test_write_string():
    w = ReprWriter()
    w.write("hello")
    s = w.string()
    assert s == "hello"


def test_writeln_string():
    w = ReprWriter()
    w.writeln("hello")
    s = w.string()
    assert s == "hello\n"


def test_indent():
    w = ReprWriter()
    w.writeln("hello {")
    with w.indent():
        w.writeln("bob")
    w.write("}")
    s = w.string()
    assert s == "hello {\n  bob\n}"


def test_indent_many():
    w = ReprWriter()
    w.writeln("hello {")
    with w.indent():
        w.writeln("bob\nchips")
    w.write("}")
    s = w.string()
    assert s == "hello {\n  bob\n  chips\n}"


def test_tag():
    s = object_repr("HelloWorld").string()
    assert s == "<HelloWorld>"


def test_comment():
    s = object_repr("HelloWorld", comment="10 entities").string()
    assert s == "<HelloWorld (10 entities)>"


def test_words():
    s = object_repr("Hello", "world", comment="10 entities").string()
    assert s == "<Hello world (10 entities)>"


def test_attrs():
    s = object_repr("Hello", "world", entities=10).string()
    assert s == "<Hello world {\n  entities: 10\n}>"


def test_nested_repr():
    box = MsgBox("readme")
    s = object_repr("Hello", "world", box=box).string()
    assert s == "<Hello world {\n  box: <MsgBox readme>\n}>"
