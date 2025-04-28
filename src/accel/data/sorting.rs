use std::cmp::Ordering;

use arrow::{
    array::RecordBatch,
    compute::{LexicographicalComparator, SortColumn},
    pyarrow::PyArrowType,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

#[pyfunction]
pub(super) fn is_strictly_sorted<'py>(
    data: Vec<PyArrowType<RecordBatch>>,
    columns: Vec<String>,
) -> PyResult<bool> {
    let lengths: Vec<_> = data.iter().map(|rb| rb.0.num_rows()).collect();
    let comps: Vec<_> = data
        .into_iter()
        .map(|pat| make_lexicomp(pat.0, &columns))
        .collect();

    // FIXME: cannot currently compare across the boundaries of two arrays
    for (i, comp) in comps.into_iter().enumerate() {
        let comp = comp?;
        for j in 1..lengths[i] {
            match comp.compare(j - 1, j) {
                // j-1 < j, we're good
                Ordering::Less => (),
                // oops
                _ => return Ok(false),
            }
        }
    }

    Ok(true)
}

fn make_lexicomp(batch: RecordBatch, columns: &[String]) -> PyResult<LexicographicalComparator> {
    let mut cols = Vec::with_capacity(columns.len());
    for ci in 0..columns.len() {
        let rbc = batch
            .column_by_name(&columns[ci])
            .ok_or_else(|| PyValueError::new_err(format!("unknown column: {}", columns[ci])))?;
        cols.push(SortColumn {
            values: rbc.clone(),
            options: None,
        });
    }
    LexicographicalComparator::try_new(&cols)
        .map_err(|e| PyRuntimeError::new_err(format!("comparison setup failed: {:?}", e)))
}
