// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Atomic utilities

use std::{
    hint::spin_loop,
    mem, ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

/// A cell supporting atomic mutation with spin locks for waiters.
///
/// This uses spin loops on atomic pointers instead of full OS-level mutexes, so
/// we can create a lot of these without incurring the overhead of going to
/// pthreads.
#[repr(transparent)]
pub struct AtomicCell<T: Send + Sync> {
    pointer: AtomicPtr<T>,
}

impl<T: Send + Sync> AtomicCell<T> {
    /// Construct a new atomic cell.
    pub fn new(value: T) -> Self {
        let ptr = Box::leak(Box::new(value));
        AtomicCell {
            pointer: AtomicPtr::new(ptr),
        }
    }

    /// Create a new vector of atomic cells.
    pub fn new_vec<S: IntoIterator<Item = T>>(src: S) -> Vec<AtomicCell<T>> {
        src.into_iter().map(AtomicCell::new).collect()
    }

    /// Mutate the cell.
    pub fn update<R, F: FnMut(&mut T) -> R>(&self, mut func: F) -> R {
        let ptr = self.acquire();
        let tref = unsafe { &mut *ptr };
        let res = func(tref);
        self.assign(ptr);
        res
    }

    /// Acquire the pointer.
    fn acquire(&self) -> *mut T {
        // load the pointer
        let mut ptr = self.pointer.load(Ordering::Acquire);
        loop {
            if ptr.is_null() {
                // the cell is locked, loop and try again
                spin_loop();
                ptr = self.pointer.load(Ordering::Acquire);
            } else {
                // the cell is unlocked, try to take the lock
                let null = ptr::null_mut::<T>();
                match self.pointer.compare_exchange_weak(
                    ptr,
                    null,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // we successfully took the lock, return the pointer
                        return ptr;
                    }
                    Err(p) => {
                        // someone else updated, so we need to try again
                        ptr = p;
                    }
                }
            }
        }
    }

    /// Set the pointer.
    fn assign(&self, ptr: *mut T) {
        // just make sure we aren't locked
        let cur = self.pointer.load(Ordering::Relaxed);
        assert!(cur.is_null());
        loop {
            match self
                .pointer
                .compare_exchange_weak(cur, ptr, Ordering::Release, Ordering::Relaxed)
            {
                Ok(_) => {
                    return;
                }
                Err(p) if !p.is_null() => {
                    eprintln!("thread wrote, bailing");
                    panic!("another thread wrote while we were working");
                }
                _ => (),
            }
        }
    }
}

impl<T: Default + Send + Sync> AtomicCell<T> {
    pub fn unwrap_vec<S: IntoIterator<Item = AtomicCell<T>>>(src: S) -> Vec<T> {
        src.into_iter()
            .map(|c| {
                let ptr = c.acquire();
                let obj = unsafe { &mut *ptr };
                mem::take(obj)
            })
            .collect()
    }
}

impl<T: Send + Sync> Drop for AtomicCell<T> {
    fn drop(&mut self) {
        let ptr = self.pointer.get_mut();
        if !ptr.is_null() {
            let _ = unsafe { Box::from_raw(ptr) };
        }
    }
}
