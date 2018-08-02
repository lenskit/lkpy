import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_blas as blas

from libc.stdlib cimport malloc, free

cdef struct AccHeap:
    int nmax
    int size
    int* keys
    double* values

cdef AccHeap* ah_create(double* values, int nmax) nogil:
    """
    Create an accumulation heap with a limited size.

    Args:
        values: the values (not owned by this heap)
        nmax: the maximum number of keys to retain
    """
    cdef AccHeap* heap = <AccHeap*> malloc(sizeof(AccHeap))
    heap.nmax = nmax
    heap.size = 0
    heap.values = values
    heap.keys = <int*> malloc(sizeof(int) * (nmax + 1))
    return heap


cdef void ah_free(AccHeap* heap) nogil:
    free(heap.keys)
    free(heap)

cdef void ah_add(AccHeap* heap, int key) nogil:
    heap.keys[heap.size] = key
    ind_upheap(heap.size, heap.keys, heap.values)
    if heap.size < heap.nmax:
        heap.size = heap.size + 1
    else:
        # we are at capacity, we need to drop the smallest value
        heap.keys[0] = heap.keys[heap.size]
        ind_downheap(0, heap.size, heap.keys, heap.values)

cdef int ah_remove(AccHeap* heap) nogil:
    cdef int top = heap.keys[0]
    if heap.size == 0:
        return -1

    heap.keys[0] = heap.keys[heap.size - 1]
    heap.size = heap.size - 1
    if heap.size > 0:
        ind_downheap(0, heap.size, heap.keys, heap.values)
    return top


cdef class Accumulator:
    cdef np.float_t[::1] values
    cdef AccHeap* heap

    def __cinit__(self, np.float_t[::1] values, int nmax):
        self.values = values
        if values.shape[0] > 0:
            self.heap = ah_create(&values[0], nmax)
        else:
            self.heap = ah_create(NULL, nmax)
    
    def __dealloc__(self):
        ah_free(self.heap)

    def __len__(self):
        return self.heap.size
    
    cpdef add(self, int key):
        if key < 0 or key >= self.values.shape[0]:
            raise IndexError()
        ah_add(self.heap, key)
    
    cpdef int peek(self):
        if self.heap.size > 0:
            return self.heap.keys[0]
        else:
            return -1

    cpdef int remove(self):
        return ah_remove(self.heap)


cdef void ind_upheap(int pos, int* keys, double* values) nogil:
    cdef int current, parent, kt
    current = pos
    parent = (current - 1) // 2
    while current > 0 and values[keys[parent]] > values[keys[current]]:
        # swap up
        kt = keys[parent]
        keys[parent] = keys[current]
        keys[current] = kt
        current = parent
        parent = (current - 1) // 2

cdef void ind_downheap(int pos, int len, int* keys, double* values) nogil:
    cdef int left, right, min, kt
    min = pos
    left = 2*pos + 1
    right = 2*pos + 2
    if left < len and values[keys[left]] < values[keys[min]]:
        min = left
    if right < len and values[keys[right]] < values[keys[min]]:
        min = right
    if min != pos:
        kt = keys[min]
        keys[min] = keys[pos]
        keys[pos] = kt
        ind_downheap(min, len, keys, values)


cdef void zero(double* vals, int n) nogil:
    for i in range(n):
        vals[i] = 0

cpdef zero_buf(double[::1] buf):
    if buf.shape[0] > 0:
        zero(&buf[0], buf.shape[0])
