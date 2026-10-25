// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crate::tasks::CancelImpl;

pub struct AtomicCancel {
    total: AtomicUsize,
    cancelled: AtomicBool,
}

impl AtomicCancel {
    pub fn new() -> Arc<AtomicCancel> {
        Arc::new(AtomicCancel {
            total: AtomicUsize::new(0),
            cancelled: AtomicBool::new(false),
        })
    }

    pub fn advance(&self, incr: usize) {
        self.total.fetch_add(incr, Ordering::Relaxed);
    }
}

impl CancelImpl for Arc<AtomicCancel> {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn current_progress(&self) -> Option<(u64, Option<u64>)> {
        let n = self.total.load(Ordering::Relaxed);
        Some((n as u64, None))
    }
}
