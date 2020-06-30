#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <mkl_spblas.h>

#ifdef _WIN32
#define EXPORT __declspec( dllexport )
#else
#define EXPORT
#endif

#define H(p) ((lk_mh_t) (p))
#define MP(h) ((sparse_matrix_t) (h))

#include "mkl_ops.h"

void check_return(const char *call, sparse_status_t rc)
{
    const char *message = "unknown";
    switch(rc) {
    case SPARSE_STATUS_SUCCESS:
        return;
    case SPARSE_STATUS_NOT_INITIALIZED:
        message = "not-initialized";
        break;
    case SPARSE_STATUS_ALLOC_FAILED:
        message = "alloc-failed";
        break;
    case SPARSE_STATUS_INVALID_VALUE:
        message = "invalid-value";
        break;
    case SPARSE_STATUS_EXECUTION_FAILED:
        message = "execution-failed";
        break;
    case SPARSE_STATUS_INTERNAL_ERROR:
        message = "internal-error";
        break;
    case SPARSE_STATUS_NOT_SUPPORTED:
        message = "not-supported";
        break;
    }
    fprintf(stderr, "MKL call %s failed with code %d (%s)\n", call, rc, message);
    abort();
}

EXPORT lk_mh_t
lk_mkl_spcreate(int nrows, int ncols, int *rowptrs, int *colinds, double *values)
{
    sparse_matrix_t matrix = NULL;
    sparse_status_t rv;

    rv = mkl_sparse_d_create_csr(&matrix, SPARSE_INDEX_BASE_ZERO, nrows, ncols, 
                                 rowptrs, rowptrs + 1, colinds, values);
    check_return("mkl_sparse_d_create_csr", rv);
    
#ifdef LK_TRACE
    fprintf(stderr, "allocated 0x%8lx (%dx%d)\n", matrix, nrows, ncols);
#endif
    return H(matrix);
}

EXPORT lk_mh_t
lk_mkl_spsubset(int rsp, int rep, int ncols, int *rowptrs, int *colinds, double *values)
{
    sparse_matrix_t matrix = NULL;
    sparse_status_t rv;
    int nrows = rep - rsp;

    rv = mkl_sparse_d_create_csr(&matrix, SPARSE_INDEX_BASE_ZERO, nrows, ncols,
                                 rowptrs + rsp, rowptrs + rsp + 1, colinds, values);
    check_return("mkl_sparse_d_create_csr", rv);

#ifdef LK_TRACE
    fprintf(stderr, "allocated 0x%8lx (%d:%d)x%d\n", matrix, rsp, rep, ncols);
#endif
    return H(matrix);
}

EXPORT int
lk_mkl_spfree(lk_mh_t matrix)
{
    sparse_status_t rv;
#ifdef LK_TRACE
    fprintf(stderr, "destroying 0x%8lx\n", matrix);
#endif
    rv = mkl_sparse_destroy(MP(matrix));
    check_return("mkl_sparse_destroy", rv);
    return rv;
}

EXPORT int
lk_mkl_sporder(lk_mh_t matrix)
{
    sparse_status_t rv;
#ifdef LK_TRACE
    fprintf(stderr, "ordering 0x%8lx\n", matrix);
#endif
    rv = mkl_sparse_order(MP(matrix));
    check_return("mkl_sparse_order", rv);
    return rv;
}

EXPORT int
lk_mkl_spopt(lk_mh_t matrix)
{
    sparse_status_t rv;
#ifdef LK_TRACE
    fprintf(stderr, "optimizing 0x%8lx\n", matrix);
#endif
    rv = mkl_sparse_optimize(MP(matrix));
    check_return("mkl_sparse_optimize", rv);
    return rv;
}

EXPORT int
lk_mkl_spmv(double alpha, lk_mh_t matrix, double *x, double beta, double *y)
{
    struct matrix_descr descr = {
        SPARSE_MATRIX_TYPE_GENERAL, 0, 0
    };
    sparse_status_t rv;
    rv = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, MP(matrix), descr, x, beta, y);
    check_return("mkl_sparse_d_mv", rv);
    return rv;
}

/**
 * Compute A * B
 */
EXPORT lk_mh_t
lk_mkl_spmab(lk_mh_t a, lk_mh_t b)
{
    sparse_matrix_t c = NULL;
    sparse_status_t rv;
    
#ifdef LK_TRACE
    fprintf(stderr, "multiplying 0x%8lx x 0x%8lx", a, b);
#endif
    rv = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, MP(a), MP(b), &c);
#ifdef LK_TRACE
    fprintf(stderr, " -> 0x%8lx\n", c);
#endif
    check_return("mkl_sparse_spmm", rv);
    
    return H(c);
}

/**
 * Compute A * B^T
 */
EXPORT lk_mh_t
lk_mkl_spmabt(lk_mh_t a, lk_mh_t b)
{
    sparse_matrix_t c = NULL;
    sparse_status_t rv;
    struct matrix_descr descr = {
        SPARSE_MATRIX_TYPE_GENERAL, 0, 0
    };

    rv = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, descr, MP(a),
                         SPARSE_OPERATION_TRANSPOSE, descr, MP(b),
                         SPARSE_STAGE_FULL_MULT, &c);
#ifdef LK_TRACE
    fprintf(stderr, "mult 0x%8lx x 0x%8lx^T -> 0x%8lx\n", a, b, c);
#endif
    check_return("mkl_sparse_sp2m", rv);
    
    return H(c);
}

EXPORT struct lk_csr
lk_mkl_spexport(lk_mh_t matrix)
{
    struct lk_csr csr;
    sparse_status_t rv;
    sparse_index_base_t idx;

#ifdef LK_TRACE
    fprintf(stderr, "export 0x%8lx\n", matrix);
#endif
    rv = mkl_sparse_d_export_csr(MP(matrix), &idx, &csr.nrows, &csr.ncols,
                                 &csr.row_sp, &csr.row_ep, &csr.colinds, &csr.values);
    
    check_return("mkl_sparse_d_export_csr", rv);
    
    return csr;
}


/* Pointer-based export interface for Numba. */
EXPORT void* lk_mkl_spexport_p(lk_mh_t matrix)
{
    struct lk_csr *ep = malloc(sizeof(struct lk_csr));
    if (!ep) return NULL;

    *ep = lk_mkl_spexport(matrix);
    return ep;
}

EXPORT void lk_mkl_spe_free(void* ep)
{
    free(ep);
}

EXPORT int lk_mkl_spe_nrows(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->nrows;
}
EXPORT int lk_mkl_spe_ncols(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->ncols;
}
EXPORT int* lk_mkl_spe_row_sp(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->row_sp;
}
EXPORT int* lk_mkl_spe_row_ep(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->row_ep;
}
EXPORT int* lk_mkl_spe_colinds(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->colinds;
}
EXPORT double* lk_mkl_spe_values(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->values;
}
