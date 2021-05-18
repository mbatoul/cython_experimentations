# cython: boundscheck=False

cimport numpy as np
import numpy as np
from libc.math cimport fabs
from cython.parallel cimport prange
from cython cimport floating, integral
cimport cython

cdef void _compute_dist(
    floating[::1] X_a_row,
    floating[::1] X_b_row,
    integral n_features,
    floating *dist,
) nogil:
    cdef integral i
    
    for i in range(n_features):
        dist[0] += fabs(X_a_row[i] - X_b_row[i])

cdef void _pairwise_dist(
    floating[:, ::1] X_a, # IN
    floating[:, ::1] X_b, # IN
    floating[:, ::1] distances, # OUT
    integral dummy
) nogil:
    cdef:
        integral i, j
        integral n_rows_X_a = X_a.shape[0]
        integral n_rows_X_b = X_b.shape[0]
        integral n_features = X_a.shape[1]
        
    for i in prange(n_rows_X_a, nogil=True):
        for j in range(n_rows_X_b):
            _compute_dist(X_a[i], X_b[j], n_features, &distances[i, j])

def pairwise_dist(
    floating[:, ::1] X_a,
    floating[:, ::1] X_b
):
    float_dtype = np.float32 if floating is float else np.float64
    cdef:
        floating[:, ::1] distances = np.zeros([X_a.shape[0], X_b.shape[0]], dtype=float_dtype)
    
    _pairwise_dist(X_a, X_b, distances, 42)
    
    return np.asarray(distances)