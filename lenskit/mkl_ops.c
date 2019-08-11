#include <stdlib.h>

#include <mkl_spblas.h>

// #ifdef WIN32
#define EXPORT __declspec( dllexport )
// #else
// #define EXPORT
// #endif

#include "mkl_ops.h"


EXPORT void*
lk_mkl_spcreate(int nrows, int ncols, int *rowptrs, int *colinds, double *values)
{
    sparse_matrix_t matrix;
    sparse_status_t rv;

    rv = mkl_sparse_d_create_csr(&matrix, SPARSE_INDEX_BASE_ZERO, nrows, ncols, 
                                 rowptrs, rowptrs + 1, colinds, values);
    if (rv) {
        return NULL;
    } else {
        return matrix;
    }
}

EXPORT int
lk_mkl_spfree(void *matrix)
{
    return mkl_sparse_destroy(matrix);
}

EXPORT int
lk_mkl_sporder(void* matrix)
{
    mkl_sparse_order(matrix);
}

EXPORT int
lk_mkl_spmv(double alpha, void* matrix, double *x, double beta, double *y)
{
    struct matrix_descr descr = {
        SPARSE_MATRIX_TYPE_GENERAL, 0, 0
    };
    return mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, matrix, descr, x, beta, y);
}

EXPORT struct lk_csr
lk_mkl_spexport(void* matrix)
{
    struct lk_csr csr;
    sparse_status_t rv;
    sparse_index_base_t idx;

    rv = mkl_sparse_d_export_csr(matrix, &idx, &csr.nrows, &csr.ncols,
                                 &csr.row_sp, &csr.row_ep, &csr.colinds, &csr.values);
    if (rv) {
        csr.nrows = -1;
        csr.ncols = -1;
    }
    return csr;
}

EXPORT void*
lk_mkl_spsyrk(void* matrix)
{
    sparse_matrix_t out;
    sparse_status_t rv;

    rv = mkl_sparse_syrk(SPARSE_OPERATION_TRANSPOSE, matrix, &out);

    if (rv) {
        return NULL;
    } else {
        return out;
    }
}
