# cython: language_level=3str, initializedcheck=False
cimport cython

cdef class KVPHeap:
    cdef readonly int sp, ep, lim
    cdef int[::1] keys
    cdef double[::1] vals

    def __cinit__(self, int sp, int ep, int lim, int[::1] keys, double[::1] vals):
        if ep < sp:
            raise ValueError("ep before sp")
        if ep - sp > lim:
            raise ValueError("array already exceeds limit")
        if sp + lim > keys.shape[0]:
            raise ValueError("key array too short")
        if sp + lim > vals.shape[0]:
            raise ValueError("value array too short")

        self.sp = sp
        self.ep = ep
        self.lim = lim
        self.keys = keys
        self.vals = vals

    cpdef int insert(self, int k, double v) except -1:
        if self.ep - self.sp < self.lim:
            # insert into heap without size problems
            # put on end, then upheap
            self.keys[self.ep] = k
            self.vals[self.ep] = v
            self._upheap()
            self.ep = self.ep + 1
            return self.ep

        elif v > self.vals[self.sp]:
            # heap is full, but new value is larger than old min
            # stick it on the front, and downheap
            self.keys[self.sp] = k
            self.vals[self.sp] = v
            self._downheap(self.lim)
            return self.ep

        else:
            # heap is full and new value doesn't belong
            return self.ep


    cpdef void sort(self):
        cdef int i = self.ep - self.sp - 1
        while i > 0:
            self._swap(i, 0)
            self._downheap(i)
            i -= 1


    cdef void _downheap(self, int limit) noexcept nogil:
        cdef bint finished = False
        cdef int pos = 0
        cdef int min, left, right
        while not finished:
            min = pos
            left = 2 * pos + 1
            right = 2 * pos + 2
            if left < limit and self._val(left) < self._val(min):
                min = left
            if right < limit and self._val(right) < self._val(min):
                min = right
            if min != pos:
                # we want to swap!
                self._swap(pos, min)
                pos = min
            else:
                finished = True


    cdef void _upheap(self) noexcept nogil:
        cdef int pos = self.ep - self.sp
        cdef int parent = (pos - 1) // 2
        while pos > 0 and self._val(parent) > self._val(pos):
            self._swap(parent, pos)
            pos = parent
            parent = (pos - 1) // 2


    cdef int _offset(self, int i) noexcept nogil:
        return self.sp + i


    cdef void _swap(self, int i1, int i2) noexcept nogil:
        cdef int p1 = self._offset(i1)
        cdef int p2 = self._offset(i2)
        cdef int tk
        cdef double tv

        tk = self.keys[p1]
        self.keys[p1] = self.keys[p2]
        self.keys[p2] = tk

        tv = self.vals[p1]
        self.vals[p1] = self.vals[p2]
        self.vals[p2] = tv


    cdef int _key(self, int i) noexcept nogil:
        return self.keys[self._offset(i)]


    cdef double _val(self, int i) noexcept nogil:
        return self.vals[self._offset(i)]
