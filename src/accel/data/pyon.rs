// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for reading invalid JSON that is actually valid Python expression syntax.

use std::borrow::Cow;

use log::*;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyNone, PyString},
};

use serde_json::{Number, Value};

/// Parse a “pyson” object.
#[pyfunction]
pub fn pyon_loads<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    match pyon_parser::expr(text) {
        Ok(ast) => {
            let obj = realize_value(py, ast)?;
            Ok(obj)
        }
        Err(e) => {
            error!(
                "parse error at {}:{}: found {}, expected {:}",
                e.location.line,
                e.location.column,
                text.chars()
                    .nth(e.location.offset)
                    .map(|c| format!("“{}”", c))
                    .unwrap_or("<EOF>".into()),
                e.expected
            );
            Err(PyValueError::new_err(format!("parse error: {:}", e)))
        }
    }
}

peg::parser! {
    grammar pyon_parser() for str {
        rule _ = quiet!{[' ' | '\n' | '\t' | '\r' | '\n']*}

        rule digit() = ['0'..='9']
        rule digits() -> &'input str
        = $(digit()+)

        rule number() -> Value
        = e:$("-"? digits() "." digits()) {Value::Number(Number::from_f64(e.parse::<f64>().unwrap()).unwrap())}
        / e:$("-" digits()) {Value::Number(e.parse::<i64>().unwrap().into())}
        / e:digits() {Value::Number(e.parse::<u64>().unwrap().into())}

        rule boolean() -> Value
        = ("true" / "True") { Value::Bool(true)}
        / ("false" / "False") { Value::Bool(false)}

        rule null() -> Value
        = ("null" / "None") { Value::Null}

        rule _string() -> String
        = "b"? "'" parts:((_str_part_sq() / echar())*) "'" {parts.into_iter().collect()}
        / "b"? "\"" parts:((_str_part_dq() / echar())*) "\"" {parts.into_iter().collect()}

        rule _str_part_sq() -> Cow<'input, str>
        = s:$([^'\'' | '\\']+) {Cow::Borrowed(s)}

        rule _str_part_dq() -> Cow<'input, str>
        = s:$([^'"' | '\\']+) {Cow::Borrowed(s)}

        rule string() -> Value
        = s:_string() {Value::String(s)}

        rule echar() -> Cow<'input, str>
        = "\\t" {"\t".into()}
        / "\\r" {"\r".into()}
        / "\\n" {"\n".into()}
        / "\\u" x:$(['a'..='f'| 'A'..='F'| '0'..='9']*<4,4>) {
            let s = format!("0x{}", x);
            let c = u32::from_str_radix(x, 16).expect("invalid hex");
            String::from_iter([char::from_u32(c).expect("invalid cahracter")]).into()
        }
        / "\\U" x:$(['a'..='f'| 'A'..='F'| '0'..='9']*<8,8>) {
            let s = format!("0x{}", x);
            let c = u32::from_str_radix(x, 16).expect("invalid hex");
            String::from_iter([char::from_u32(c).expect("invalid cahracter")]).into()
        }
        / "\\" c:$([_]) {c.into()}

        rule list() -> Value
        = "[" e:(expr() ** ",") _ ","? "]" {Value::Array(e)}

        rule object() -> Value
        = "{" entries:(object_entry() ** ",") _ ","? "}" {Value::Object(entries.into_iter().collect())}

        rule object_entry() -> (String, Value)
        = _ k:_string() _ ":" v:expr() {(k, v)}

        rule _expr() -> Value
        = null()
        / boolean()
        / number()
        / string()
        / list()
        / object()

        pub rule expr() -> Value = _ e:_expr() _ {e}
}
}

fn realize_value<'py>(py: Python<'py>, ast: Value) -> PyResult<Bound<'py, PyAny>> {
    match ast {
        Value::Null => Ok(PyNone::get(py).to_owned().into_any()),
        Value::Bool(val) => Ok(PyBool::new(py, val).to_owned().into_any()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(PyInt::new(py, i).into_any())
            } else if let Some(x) = n.as_f64() {
                Ok(PyFloat::new(py, x).into_any())
            } else {
                Err(PyValueError::new_err(format!("invalid number {:?}", n)))
            }
        }
        Value::String(s) => Ok(PyString::new(py, &s).into_any()),
        Value::Array(list) => {
            let out = PyList::empty(py);
            for elt in list {
                out.append(realize_value(py, elt)?)?;
            }

            Ok(out.into_any())
        }
        Value::Object(dict) => {
            let out = PyDict::new(py);

            for (k, v) in dict {
                out.set_item(k, realize_value(py, v)?)?;
            }

            Ok(out.into_any())
        }
    }
}
