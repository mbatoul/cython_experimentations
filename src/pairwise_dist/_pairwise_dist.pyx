# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=0

cimport numpy as np
import numpy as np
from libc.math cimport fabs
from cython.parallel cimport prange
from cython cimport floating, integral
cimport cython

cdef floating _l1_distance(
    floating[::1] X_a_row,
    floating[::1] X_b_row,
    integral n_features,
) nogil:
    cdef:
        int i
        floating dist = 0
    
    for i in range(n_features):
        dist += fabs(X_a_row[i] - X_b_row[i])
        
    return dist

cdef void _pairwise_dist(
    floating[:, ::1] X_a, # IN
    floating[:, ::1] X_b, # IN
    floating[:, ::1] distances, # OUT
) nogil:
    cdef:
        int i, j
        int n_rows_X_a = X_a.shape[0]
        int n_rows_X_b = X_b.shape[0]
        int n_features = X_a.shape[1]
        
    for i in prange(n_rows_X_a, nogil=True):
        for j in range(n_rows_X_b):
            distances[i, j] = _l1_distance(X_a[i], X_b[j], n_features)

def pairwise_dist(
    floating[:, ::1] X_a,
    floating[:, ::1] X_b
):
    float_dtype = np.float32 if floating is float else np.float64
    cdef:
        floating[:, ::1] distances = np.zeros([X_a.shape[0], X_b.shape[0]], dtype=float_dtype)
    
    _pairwise_dist(X_a, X_b, distances)
    
    return np.asarray(distances)