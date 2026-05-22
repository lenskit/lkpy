// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Arrow list support code.

use arrow::{
    array::{Array, LargeListArray, ListArray},
    buffer::{OffsetBuffer, ScalarBuffer},
};

use log::*;

/// Trait for converting to a large list array.
///
/// This trait is to allow code to generalize over list types, and obtain a large
/// list for consistency.
pub trait ExtractListArray: Sized {
    /// Get the array as a large list array.
    fn extract_list_array(array: &dyn Array) -> Option<Self>;
}

impl ExtractListArray for LargeListArray {
    fn extract_list_array(array: &dyn Array) -> Option<Self> {
        let any = array.as_any();
        if let Some(arr) = any.downcast_ref::<LargeListArray>() {
            Some(arr.clone())
        } else if let Some(arr) = any.downcast_ref::<ListArray>() {
            info!("converting type {}", arr.data_type());
            let (field, offsets, values, nulls) = arr.clone().into_parts();
            info!("field: {}", field);
            // convert offsets into Int64
            let offsets: Vec<_> = offsets.iter().map(|o| *o as i64).collect();
            let offsets = ScalarBuffer::from(offsets);
            let offsets = OffsetBuffer::new(offsets);
            Some(
                LargeListArray::try_new(field, offsets, values, nulls)
                    .expect("array conversion failed"),
            )
        } else {
            None
        }
    }
}

impl ExtractListArray for ListArray {
    fn extract_list_array(array: &dyn Array) -> Option<Self> {
        array.as_any().downcast_ref::<Self>().map(Clone::clone)
    }
}
