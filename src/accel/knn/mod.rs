// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use pyo3::prelude::*;

mod accum;
mod item_score;
mod item_train;
mod slim;
mod user_score;

#[pymodule(name = "knn")]
pub mod pymod {
    #[pymodule_export]
    use super::item_score::score_explicit;
    #[pymodule_export]
    use super::item_score::score_implicit;
    #[pymodule_export]
    use super::item_train::compute_similarities;
    #[pymodule_export]
    use super::slim::train_slim;
    #[pymodule_export]
    use super::user_score::user_score_items_explicit;
    #[pymodule_export]
    use super::user_score::user_score_items_implicit;
}
