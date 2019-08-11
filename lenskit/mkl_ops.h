struct lk_csr {
    int nrows;
    int ncols;
    int *row_sp;
    int *row_ep;
    int *colinds;
    double *values;
};

EXPORT void* lk_mkl_spcreate(int nrows, int ncols, int *rowptrs, int *colinds, double *values);
EXPORT int lk_mkl_spfree(void *matrix);
EXPORT int lk_mkl_sporder(void* matrix);
EXPORT int lk_mkl_spmv(double alpha, void* matrix, double *x, double beta, double *y);
EXPORT struct lk_csr lk_mkl_spexport(void* matrix);
EXPORT void* lk_mkl_spsyrk(void* matrix);
