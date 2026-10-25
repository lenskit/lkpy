// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Tasks that report progress.

use rayon::iter::ParallelIterator;
use rayon_cancel::{CancelAdapter, CancelHandle, CountHandle};

use crate::tasks::CancelImpl;

pub struct IterCancel {
    cancel: CancelHandle,
    counter: CountHandle,
}

impl CancelImpl for IterCancel {
    fn cancel(&self) {
        self.cancel.cancel();
    }

    fn current_progress(&self) -> Option<(u64, Option<u64>)> {
        Some((self.counter.get() as u64, None))
    }
}

impl IterCancel {
    pub fn from_adapter<I: ParallelIterator>(adapter: &CancelAdapter<I>) -> Self {
        let cancel = adapter.canceller();
        let counter = adapter.counter();
        IterCancel { cancel, counter }
    }
}
