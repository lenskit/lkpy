//! Accelerated sampling support.
use std::mem;

use arrow::{
    array::{downcast_array, make_array, Array, ArrayData, Int32Array},
    pyarrow::PyArrowType,
};
use arrow_schema::DataType;
use log::*;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
};

use crate::data::RowColumnSet;
use crate::types::checked_array_convert;

#[pyclass]
pub struct NegativeSampler {
    rc_set: Py<RowColumnSet>,
    rows: Int32Array,
    n_cols: usize,
    negatives: Vec<i32>,
    remaining: Vec<u32>,
}

impl NegativeSampler {
    fn element_row(&self, index: usize) -> usize {
        index / self.n_cols
    }

    fn element_src_row(&self, index: usize) -> i32 {
        let row = self.element_row(index);
        self.rows.value(row)
    }
}

#[pymethods]
impl NegativeSampler {
    #[new]
    fn new<'py>(
        rc_set: Bound<'py, RowColumnSet>,
        users: PyArrowType<ArrayData>,
        tgt_n: usize,
    ) -> PyResult<Self> {
        let users = make_array(users.0);
        if users.data_type() != &DataType::Int32 {
            return Err(PyTypeError::new_err(format!(
                "unexpected user type {} (expected int32)",
                users.data_type()
            )));
        }

        let n_rows = users.len();
        let n = n_rows * tgt_n;
        debug!(
            "creating sampler for {} negatives for {} rows",
            tgt_n, n_rows
        );

        Ok(NegativeSampler {
            rc_set: rc_set.unbind(),
            rows: downcast_array(&users),
            n_cols: tgt_n,
            negatives: vec![-1; n],
            remaining: (0..n as u32).collect(),
        })
    }

    fn num_remaining(&self) -> usize {
        self.remaining.len()
    }

    fn accumulate<'py>(
        &mut self,
        py: Python<'py>,
        items: PyArrowType<ArrayData>,
        force: bool,
    ) -> PyResult<()> {
        if self.negatives.is_empty() {
            return Err(PyRuntimeError::new_err(
                "sampler already finished".to_string(),
            ));
        }
        let items = make_array(items.0);
        let iref: &Int32Array = checked_array_convert("items", "int32", &items)?;
        let rcs_ref = self.rc_set.borrow(py);

        let nr = self.remaining.len();
        let mut remaining = Vec::with_capacity(nr);

        for i in 0..nr {
            let pos = self.remaining[i] as usize;
            let item = iref.value(i);
            let user = self.element_src_row(pos);
            if force || !rcs_ref.contains_pair(user, item) {
                self.negatives[pos] = item;
            } else {
                remaining.push(pos as u32);
            }
        }

        mem::swap(&mut remaining, &mut self.remaining);

        Ok(())
    }

    fn result(&mut self) -> PyResult<PyArrowType<ArrayData>> {
        let array = mem::take(&mut self.negatives);
        let array = Int32Array::from(array);
        Ok(array.into_data().into())
    }
}
