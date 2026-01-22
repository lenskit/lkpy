// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::{collections::LinkedList, sync::Arc};

use arrow::{
    array::{Float32Builder, Int32Builder, LargeListArray, StructArray},
    buffer::OffsetBuffer,
};
use arrow_schema::{DataType, Field, Fields};
use pyo3::prelude::*;
use rayon::iter::plumbing::{Consumer, Folder, Reducer};

use crate::progress::ProgressHandle;

use super::SparseIndexType;

pub type CSRItem = Vec<(i32, f32)>;
pub type CSRResult = LinkedList<LargeListArray>;

/// Rayon consumer that collects sparse rows (as vectors) into a chunked sparse row array.
///
/// It handles checking for signals and updating progress.
pub struct ArrowCSRConsumer {
    state: Arc<CSRState>,
    lengths: Vec<usize>,
    col_bld: Int32Builder,
    val_bld: Float32Builder,
}

struct CSRState {
    dimension: usize,
    progress: ProgressHandle,
}

impl CSRState {
    fn new(dim: usize, progress: Option<Py<PyAny>>) -> Self {
        CSRState {
            dimension: dim,
            progress: ProgressHandle::new(progress),
        }
    }
}

impl ArrowCSRConsumer {
    fn from_state(state: CSRState) -> Self {
        Self::from_state_ref(Arc::new(state))
    }

    fn from_state_ref(state: Arc<CSRState>) -> Self {
        ArrowCSRConsumer {
            state,
            lengths: Vec::new(),
            col_bld: Int32Builder::new(),
            val_bld: Float32Builder::new(),
        }
    }
    pub(crate) fn new(dim: usize) -> Self {
        Self::from_state(CSRState::new(dim, None))
    }

    pub(crate) fn with_progress(dim: usize, progress: Py<PyAny>) -> Self {
        Self::from_state(CSRState::new(dim, Some(progress)))
    }
}

impl Consumer<CSRItem> for ArrowCSRConsumer {
    type Folder = Self;

    type Reducer = Self;

    type Result = CSRResult;

    fn split_at(self, _index: usize) -> (Self, Self, Self::Reducer) {
        let left = ArrowCSRConsumer::from_state_ref(self.state.clone());
        let right = ArrowCSRConsumer::from_state_ref(self.state.clone());
        (left, right, self)
    }

    fn into_folder(self) -> Self::Folder {
        self
    }

    fn full(&self) -> bool {
        false
    }
}

impl Folder<CSRItem> for ArrowCSRConsumer {
    type Result = CSRResult;

    fn consume(mut self, item: CSRItem) -> Self {
        let len = item.len();
        for (i, s) in item {
            self.col_bld.append_value(i);
            self.val_bld.append_value(s);
        }
        self.lengths.push(len);
        self.state.progress.tick();
        self
    }

    fn complete(mut self) -> Self::Result {
        let struct_fields = Fields::from(vec![
            Field::new("index", DataType::Int32, false)
                .with_extension_type(SparseIndexType::create(self.state.dimension)),
            Field::new("value", DataType::Float32, false),
        ]);
        let list_field = Field::new("rows", DataType::Struct(struct_fields.clone()), false);
        let sa = StructArray::new(
            struct_fields,
            vec![
                Arc::new(self.col_bld.finish()),
                Arc::new(self.val_bld.finish()),
            ],
            None,
        );
        let list = LargeListArray::new(
            Arc::new(list_field),
            OffsetBuffer::from_lengths(self.lengths),
            Arc::new(sa),
            None,
        );
        let mut result = LinkedList::new();
        result.push_back(list);
        result
    }

    fn full(&self) -> bool {
        false
    }
}

impl Reducer<CSRResult> for ArrowCSRConsumer {
    fn reduce(self, mut left: CSRResult, mut right: CSRResult) -> CSRResult {
        left.append(&mut right);
        left
    }
}
