use std::cmp::Reverse;
use std::sync::Arc;

use arrow::{
    array::{
        make_array, Array, ArrayData, Float32Builder, Int32Builder, LargeListArray, StructArray,
    },
    buffer::OffsetBuffer,
    datatypes::{DataType, Field, Fields},
    pyarrow::PyArrowType,
};
use log::*;
use ordered_float::NotNan;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::sparse::CSRMatrix;

#[pyfunction]
pub fn compute_similarities<'py>(
    py: Python<'py>,
    ui_ratings: PyArrowType<ArrayData>,
    iu_ratings: PyArrowType<ArrayData>,
    shape: (usize, usize),
    min_sim: f32,
    save_nbrs: Option<i64>,
) -> PyResult<Vec<PyArrowType<ArrayData>>> {
    let (nu, ni) = shape;

    py.allow_threads(|| {
        // extract the data
        debug!("preparing {}x{} matrix", nu, ni);
        let ui_mat = CSRMatrix::from_arrow(make_array(ui_ratings.0), nu, ni)?;
        let iu_mat = CSRMatrix::from_arrow(make_array(iu_ratings.0), ni, nu)?;
        assert_eq!(ui_mat.array.len(), nu);
        assert_eq!(iu_mat.array.len(), ni);

        // let's compute!
        let range = 0..ni;
        debug!("computing similarity rows");
        let chunks = range
            .into_par_iter()
            .map(|row| sim_row(row, &ui_mat, &iu_mat, min_sim, save_nbrs))
            .collect_vec_list();
        // count the similarities
        let n_sim = chunks
            .iter()
            .flat_map(|v| v.iter().map(|v2| v2.len()))
            .sum::<usize>();
        debug!(
            "computed {} similarities in {} matrix chunks",
            n_sim,
            chunks.len()
        );

        // now we will post-process the chunks into Arrow arrays, one per chunk.
        // on the Python side, we will combine the chunks.
        let mut rv_chunks = Vec::new();
        for chunk in chunks {
            let lengths: Vec<usize> = chunk.iter().map(Vec::len).collect();
            let n_entries = lengths.iter().sum();
            let mut col_bld = Int32Builder::with_capacity(n_entries);
            let mut val_bld = Float32Builder::with_capacity(n_entries);

            for row in chunk.into_iter() {
                for (i, s) in row {
                    col_bld.append_value(i);
                    val_bld.append_value(s);
                }
            }

            let struct_fields = Fields::from(vec![
                Field::new("index", DataType::Int32, false),
                Field::new("value", DataType::Float32, false),
            ]);
            let list_field = Field::new("rows", DataType::Struct(struct_fields.clone()), false);
            let sa = StructArray::new(
                struct_fields,
                vec![Arc::new(col_bld.finish()), Arc::new(val_bld.finish())],
                None,
            );
            let list = LargeListArray::new(
                Arc::new(list_field),
                OffsetBuffer::from_lengths(lengths),
                Arc::new(sa),
                None,
            );
            rv_chunks.push(list.into_data().into());
        }

        Ok(rv_chunks)
    })
}

fn sim_row(
    row: usize,
    ui_mat: &CSRMatrix,
    iu_mat: &CSRMatrix,
    min_sim: f32,
    save_nbrs: Option<i64>,
) -> Vec<(i32, f32)> {
    let (r_start, r_end) = iu_mat.extent(row);

    // accumulate count and inner products
    let mut counts = vec![0; ui_mat.n_cols];
    let mut dots = vec![0.0f32; ui_mat.n_cols];

    // track output slots in use
    let mut used = Vec::new();

    // loop over the users
    for i in r_start..r_end {
        let u = iu_mat.col_inds.value(i as usize);
        let r = iu_mat.values.value(i as usize);

        let (u_start, u_end) = ui_mat.extent(u as usize);
        // loop over the users' items
        for j in u_start..u_end {
            let j = j as usize;
            let other = ui_mat.col_inds.value(j) as usize;
            if other == row {
                continue;
            }
            let orate = ui_mat.values.value(j);
            if counts[other] == 0 {
                used.push(other);
            }
            counts[other] += 1;
            dots[other] += r * orate;
        }
    }

    // finish up and return!
    let mut sims: Vec<_> = used
        .into_iter()
        .filter(|i| dots[*i] >= min_sim)
        .map(|i| (i as i32, dots[i]))
        .collect();

    // truncate if needed
    if let Some(limit) = save_nbrs {
        if limit > 0 {
            // sort by value number
            sims.sort_by_key(|(_i, s)| Reverse(NotNan::new(*s).unwrap()));
            sims.truncate(limit as usize);
            sims.shrink_to_fit();
        }
    }
    // sort by column number
    sims.sort_by_key(|(i, _s)| *i);

    sims
}
