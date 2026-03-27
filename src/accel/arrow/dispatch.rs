// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Macros for dispatching Arrow operations.

/// Dispatch call based on high-level Arrow types.
#[macro_export]
macro_rules! match_array_type {
    ($arr:expr, {$($ty:ident($var:ident) => $e:expr),+$(,)?}) => {{
        let _macro_mat_dt = $arr.data_type();
        let _macro_mat_rv = None;
        $(let _macro_mat_rv = $crate::match_branches!(_macro_mat_rv, _macro_mat_dt, $ty $var, $arr, $e);)*
        _macro_mat_rv.ok_or_else(||
            pyo3::exceptions::PyTypeError::new_err(format!(
                "unsupported type {}",
                $arr.data_type()
            )))
        }}
}

/// Internal helper macro for [match_array_type].
#[macro_export]
macro_rules! ma_invoke_branch {
    ($arr: expr, $ety:ty, $var: ident, $e: expr) => {{
        let $var = $arr.as_any().downcast_ref::<$ety>().unwrap();
        let res = $e;
        Some(res)
    }};
}

/// Internal helper macro for [match_array_type].
#[macro_export]
macro_rules! match_branches {
    ($in:expr, $dt:expr, size $var:ident, $arr:expr, $e:expr) => {
        $in.or_else(|| match $dt {
            arrow_schema::DataType::Int32 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Int32Array, $var, $e)
            }
            arrow_schema::DataType::Int64 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Int64Array, $var, $e)
            }
            _ => None,
        })
    };
    ($in:expr, $dt:expr, integer $var:ident, $arr:expr, $e:expr) => {
        $in.or_else(|| match $dt {
            arrow_schema::DataType::Int8 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Int8Array, $var, $e)
            }
            arrow_schema::DataType::Int16 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Int16Array, $var, $e)
            }
            arrow_schema::DataType::Int32 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Int32Array, $var, $e)
            }
            arrow_schema::DataType::Int64 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Int64Array, $var, $e)
            }
            arrow_schema::DataType::UInt8 => {
                $crate::ma_invoke_branch!($arr, arrow::array::UInt8Array, $var, $e)
            }
            arrow_schema::DataType::UInt16 => {
                $crate::ma_invoke_branch!($arr, arrow::array::UInt16Array, $var, $e)
            }
            arrow_schema::DataType::UInt32 => {
                $crate::ma_invoke_branch!($arr, arrow::array::UInt32Array, $var, $e)
            }
            arrow_schema::DataType::UInt64 => {
                $crate::ma_invoke_branch!($arr, arrow::array::UInt64Array, $var, $e)
            }
            _ => None,
        })
    };
    ($in:expr, $dt:expr, floating $var:ident, $arr:expr, $e:expr) => {
        $in.or_else(|| match $dt {
            arrow_schema::DataType::Float16 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Float16Array, $var, $e)
            }
            arrow_schema::DataType::Float32 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Float32Array, $var, $e)
            }
            arrow_schema::DataType::Float64 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Float64Array, $var, $e)
            }
            _ => None,
        })
    };
    ($in:expr, $dt:expr, numeric $var:ident, $arr:expr, $e:expr) => {
        $in.or_else(|| match $dt {
            arrow_schema::DataType::Int8 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Int8Array, $var, $e)
            }
            arrow_schema::DataType::Int16 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Int16Array, $var, $e)
            }
            arrow_schema::DataType::Int32 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Int32Array, $var, $e)
            }
            arrow_schema::DataType::Int64 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Int64Array, $var, $e)
            }
            arrow_schema::DataType::UInt8 => {
                $crate::ma_invoke_branch!($arr, arrow::array::UInt8Array, $var, $e)
            }
            arrow_schema::DataType::UInt16 => {
                $crate::ma_invoke_branch!($arr, arrow::array::UInt16Array, $var, $e)
            }
            arrow_schema::DataType::UInt32 => {
                $crate::ma_invoke_branch!($arr, arrow::array::UInt32Array, $var, $e)
            }
            arrow_schema::DataType::UInt64 => {
                $crate::ma_invoke_branch!($arr, arrow::array::UInt64Array, $var, $e)
            }
            arrow_schema::DataType::Float16 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Float16Array, $var, $e)
            }
            arrow_schema::DataType::Float32 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Float32Array, $var, $e)
            }
            arrow_schema::DataType::Float64 => {
                $crate::ma_invoke_branch!($arr, arrow::array::Float64Array, $var, $e)
            }
            _ => None,
        })
    };
}
