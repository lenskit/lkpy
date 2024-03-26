# cython: language_level=3str

cdef class CSMatrix:
    cdef readonly int nrows, ncols, nnz
    cdef readonly int[:] rowptr
    cdef readonly int[:] colind
    cdef readonly double[:] values

    def __cinit__(self, int nr, int nc, int[:] rps, int[:] cis, double[:] vs):
        self.nrows = nr
        self.ncols = nc
        self.rowptr = rps
        self.colind = cis
        self.values = vs
        self.nnz = self.rowptr[nr]

    cpdef (int,int) row_ep(self, row):
        if row < 0 or row >= self.nrows:
            raise IndexError(f"invalid row {row} for {self.nrows}x{self.ncols} matrix")

        return self.rowptr[row], self.rowptr[row+1]
