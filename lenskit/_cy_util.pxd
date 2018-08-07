cdef struct AccHeap:
    int nmax
    int size
    int* keys
    double* values

cdef AccHeap* ah_create(double* values, int nmax) nogil
cdef void ah_free(AccHeap* heap) nogil
cdef void ah_add(AccHeap* heap, int key) nogil
cdef int ah_remove(AccHeap* heap) nogil

cdef void zero(double* vals, int n) nogil
