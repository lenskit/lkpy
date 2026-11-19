// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

mod explicit;
mod implicit;
mod solve;

use pyo3::prelude::*;

#[pymodule(name = "als")]
pub mod pymod {
    #[pymodule_export]
    use super::explicit::train_explicit_matrix;
    #[pymodule_export]
    use super::implicit::train_implicit_matrix;
}
