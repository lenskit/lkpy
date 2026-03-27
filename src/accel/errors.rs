// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

#[macro_export]
macro_rules! ok_or_pyerr {
    ($err:expr, $ety:ty, $($arg:expr),*) => {
        $err.ok_or_else(|| <$ety>::new_err(format!($($arg),*)))
    };
}
