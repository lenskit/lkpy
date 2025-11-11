// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Macros for dispatching Arrow operations.

#[macro_export]
macro_rules! dispatch_array {
    ($arr: expr, float $fproc: ident, int $iproc: ident) => (
        match $arr.data_type() {
            DataType::Float16 => Ok($fproc($arr.as_any().downcast_ref::<Float16Array>().unwrap())),
            DataType::Float32 => Ok($fproc($arr.as_any().downcast_ref::<Float32Array>().unwrap())),
            DataType::Float64 => Ok($fproc($arr.as_any().downcast_ref::<Float64Array>().unwrap())),
            DataType::Int8 => Ok($iproc($arr.as_any().downcast_ref::<Int8Array>().unwrap())),
            DataType::Int16 => Ok($iproc($arr.as_any().downcast_ref::<Int16Array>().unwrap())),
            DataType::Int32 => Ok($iproc($arr.as_any().downcast_ref::<Int32Array>().unwrap())),
            DataType::Int64 => Ok($iproc($arr.as_any().downcast_ref::<Int64Array>().unwrap())),
            DataType::UInt8 => Ok($iproc($arr.as_any().downcast_ref::<UInt8Array>().unwrap())),
            DataType::UInt16 => Ok($iproc($arr.as_any().downcast_ref::<UInt16Array>().unwrap())),
            DataType::UInt32 => Ok($iproc($arr.as_any().downcast_ref::<UInt32Array>().unwrap())),
            DataType::UInt64 => Ok($iproc($arr.as_any().downcast_ref::<UInt64Array>().unwrap())),
            _ => {
                Err(PyTypeError::new_err(format!(
                    "unsupported type {}",
                    $arr.data_type()
                )))
            }
        }
    );
 ($arr: expr, numeric $proc: ident) => (
    dispatch_array!($arr, float $proc, int $proc)
 );
 ($arr: expr, int $iproc: ident) => (
    match $arr.data_type() {
            DataType::Int8 => Ok($iproc($arr.as_any().downcast_ref::<Int8Array>().unwrap())),
            DataType::Int16 => Ok($iproc($arr.as_any().downcast_ref::<Int16Array>().unwrap())),
            DataType::Int32 => Ok($iproc($arr.as_any().downcast_ref::<Int32Array>().unwrap())),
            DataType::Int64 => Ok($iproc($arr.as_any().downcast_ref::<Int64Array>().unwrap())),
            DataType::UInt8 => Ok($iproc($arr.as_any().downcast_ref::<UInt8Array>().unwrap())),
            DataType::UInt16 => Ok($iproc($arr.as_any().downcast_ref::<UInt16Array>().unwrap())),
            DataType::UInt32 => Ok($iproc($arr.as_any().downcast_ref::<UInt32Array>().unwrap())),
            DataType::UInt64 => Ok($iproc($arr.as_any().downcast_ref::<UInt64Array>().unwrap())),
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported type {}",
                    $arr.data_type()
                )))
            }
        }
 );
 ($arr: expr, float $fproc: ident) => (
        match $arr.data_type() {
            DataType::Float16 => Ok($fproc($arr.as_any().downcast_ref::<Float16Array>().unwrap())),
            DataType::Float32 => Ok($fproc($arr.as_any().downcast_ref::<Float32Array>().unwrap())),
            DataType::Float64 => Ok($fproc($arr.as_any().downcast_ref::<Float64Array>().unwrap())),
            _ => {
                Err(PyTypeError::new_err(format!(
                    "unsupported type {}",
                    $arr.data_type()
                )))
            }
        }
    );
    ($arr: expr, int $iproc: ident, float $fproc: ident) => (
        dispatch_array!($arr, float $fproc, int $proc)
    );
}
