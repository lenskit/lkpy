typedef intptr_t lk_mh_t;
struct lk_csr {
    int nrows;
    int ncols;
    int *row_sp;
    int *row_ep;
    int *colinds;
    double *values;
};

EXPORT lk_mh_t lk_mkl_spcreate(int nrows, int ncols, int *rowptrs, int *colinds, double *values);
EXPORT lk_mh_t lk_mkl_spsubset(int rsp, int rep, int ncols, int *rowptrs, int *colinds, double *values);
EXPORT int lk_mkl_spfree(lk_mh_t matrix);
EXPORT struct lk_csr lk_mkl_spexport(lk_mh_t matrix);

EXPORT void* lk_mkl_spexport_p(lk_mh_t matrix);
EXPORT void lk_mkl_spe_free(void* ep);

EXPORT int lk_mkl_spe_nrows(void* ep);
EXPORT int lk_mkl_spe_ncols(void* ep);
EXPORT int* lk_mkl_spe_row_sp(void* ep);
EXPORT int* lk_mkl_spe_row_ep(void* ep);
EXPORT int* lk_mkl_spe_colinds(void* ep);
EXPORT double* lk_mkl_spe_values(void* ep);

EXPORT int lk_mkl_sporder(lk_mh_t matrix);
EXPORT int lk_mkl_spopt(lk_mh_t matrix);

EXPORT int lk_mkl_spmv(double alpha, lk_mh_t matrix, double *x, double beta, double *y);
EXPORT lk_mh_t lk_mkl_spmab(lk_mh_t a, lk_mh_t b);
EXPORT lk_mh_t lk_mkl_spmabt(lk_mh_t a, lk_mh_t b);
