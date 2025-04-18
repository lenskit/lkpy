//! Row-column sets for quick masking.

use std::collections::HashSet;

use arrow::{
    array::{make_array, ArrayData},
    pyarrow::PyArrowType,
};
use pyo3::prelude::*;

use crate::sparse::CSRStructure;

#[pyclass]
pub struct RowColumnSet {
    set: HashSet<(i32, i32)>,
}

impl RowColumnSet {
    pub(crate) fn contains_pair(&self, row: i32, col: i32) -> bool {
        self.set.contains(&(row, col))
    }
}

#[pymethods]
impl RowColumnSet {
    #[new]
    fn new(matrix: PyArrowType<ArrayData>) -> PyResult<Self> {
        let matrix = make_array(matrix.0);
        let matrix: CSRStructure<i32> = CSRStructure::from_arrow(matrix)?;

        let mut set = HashSet::with_capacity(matrix.nnz());

        for r in 0..matrix.len() {
            let (sp, ep) = matrix.extent(r);
            for ci in sp..ep {
                set.insert((r as i32, matrix.col_inds.value(ci as usize) as i32));
            }
        }

        Ok(RowColumnSet { set })
    }
}
