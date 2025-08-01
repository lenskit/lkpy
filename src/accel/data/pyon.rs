// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for reading invalid JSON that is actually valid Python expression syntax.

use std::any::Any;

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyComplex, PyDict, PyFloat, PyInt, PyList, PyNone, PyString},
};

use rustpython_ast::{Constant, Expr, ExprDict, ExprList};
use rustpython_parser::Parse;

/// Parse a “pyson” object.
#[pyfunction]
pub fn pyon_loads<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    let ast = ExprDict::parse(text, "internal")
        .map_err(|_e| PyErr::new::<PyValueError, _>("Python parse error"))?;
    let obj = realize_dict(py, ast)?;
    Ok(obj)
}

fn realize_value<'py>(py: Python<'py>, ast: Expr) -> PyResult<Bound<'py, PyAny>> {
    match ast {
        Expr::Constant(c) => realize_constant(py, c.value),
        Expr::List(list) => realize_list(py, list),
        Expr::Dict(dict) => realize_dict(py, dict),
        _ => Err(PyErr::new::<PyValueError, _>(format!(
            "unsupported expression type {:?}",
            ast.type_id()
        ))),
    }
}

fn realize_constant<'py>(py: Python<'py>, c: Constant) -> PyResult<Bound<'py, PyAny>> {
    match c {
        Constant::Bool(val) => Ok(PyBool::new(py, val).to_owned().into_any()),
        Constant::Bytes(val) => Ok(PyString::new(py, str::from_utf8(&val)?).into_any()),
        Constant::None => Ok(PyNone::get(py).to_owned().into_any()),
        Constant::Str(s) => Ok(PyString::new(py, &s).into_any()),
        Constant::Int(big_int) => {
            let n: i64 = big_int
                .try_into()
                .map_err(|_e| PyErr::new::<PyValueError, _>("integer out of bounds"))?;
            Ok(PyInt::new(py, n).into_any())
        }
        Constant::Tuple(constants) => {
            let list = PyList::empty(py);
            for elt in constants {
                list.append(realize_constant(py, elt)?)?;
            }
            Ok(list.into_any())
        }
        Constant::Float(x) => Ok(PyFloat::new(py, x).into_any()),
        Constant::Complex { real, imag } => Ok(PyComplex::from_doubles(py, real, imag).into_any()),
        Constant::Ellipsis => Err(PyErr::new::<PyValueError, _>("ellipsis not supported")),
    }
}

fn realize_list<'py>(py: Python<'py>, list: ExprList) -> PyResult<Bound<'py, PyAny>> {
    let out = PyList::empty(py);

    for elt in list.elts {
        out.append(realize_value(py, elt)?)?;
    }

    Ok(out.into_any())
}

fn realize_dict<'py>(py: Python<'py>, dict: ExprDict) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);

    for (key, val) in dict.keys.into_iter().zip(dict.values.into_iter()) {
        if let Some(key) = key {
            let key = expect_string(key)?;
            let val = realize_value(py, val)?;
            out.set_item(key, val)?;
        } else {
            return Err(PyErr::new::<PyValueError, _>(
                "dictionary splat not supported",
            ));
        }
    }

    Ok(out.into_any())
}

fn expect_string(val: Expr) -> PyResult<String> {
    if let Some(c) = val.constant_expr() {
        match c.value {
            Constant::Str(s) => Ok(s),
            Constant::Bytes(s) => Ok(String::from_utf8(s)?),
            _ => Err(PyErr::new::<PyTypeError, _>(format!(
                "expected string, got {:?}",
                c.kind
            ))),
        }
    } else {
        Err(PyErr::new::<PyValueError, _>("expression is not constant"))
    }
}
