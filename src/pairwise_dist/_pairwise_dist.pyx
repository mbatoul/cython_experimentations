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
from cython cimport floating
cimport cython

cdef inline floating _l1_distance(
    floating[:, ::1] X_a,
    int i,
    floating[:, ::1] X_b,
    int j,
    int n_features,
) nogil:
    cdef:
        int k
        floating dist = 0
        int unrolled_upper = (n_features // 4) * 4
    
    for k in range(0, unrolled_upper, 4):
        dist += fabs(X_a[i, k] - X_b[j, k])
        dist += fabs(X_a[i, k + 1] - X_b[j, k + 1])
        dist += fabs(X_a[i, k + 2] - X_b[j, k + 2])
        dist += fabs(X_a[i, k + 3] - X_b[j, k + 3])
    
    for k in range(unrolled_upper, n_features):
        dist += fabs(X_a[i, k] - X_b[j, k])
        
    return dist

cdef void _pairwise_dist(
    floating[:, ::1] X_a, # IN
    floating[:, ::1] X_b, # IN
    floating[:, ::1] distances, # OUT
    bint parallel=0
) nogil:
    cdef:
        int i, j
        int n_rows_X_a = X_a.shape[0]
        int n_rows_X_b = X_b.shape[0]
        int n_features = X_a.shape[1]
    
    if parallel:
        for i in prange(n_rows_X_a, nogil=True):
            for j in range(n_rows_X_b):
                distances[i, j] = _l1_distance(X_a, i, X_b, j, n_features)
    else:
        for i in range(n_rows_X_a):
            for j in range(n_rows_X_b):
                distances[i, j] = _l1_distance(X_a, i, X_b, j, n_features)

def pairwise_dist_sequential(
    floating[:, ::1] X_a,
    floating[:, ::1] X_b
):
    float_dtype = np.float32 if floating is float else np.float64
    cdef:
        floating[:, ::1] distances = np.zeros([X_a.shape[0], X_b.shape[0]], dtype=float_dtype)
    
    _pairwise_dist(X_a, X_b, distances, parallel=0)
    
    return np.asarray(distances)

def pairwise_dist_parallel(
    floating[:, ::1] X_a,
    floating[:, ::1] X_b
):
    float_dtype = np.float32 if floating is float else np.float64
    cdef:
        floating[:, ::1] distances = np.zeros([X_a.shape[0], X_b.shape[0]], dtype=float_dtype)
    
    _pairwise_dist(X_a, X_b, distances, parallel=1)
    
    return np.asarray(distances)