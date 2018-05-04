# cython: language_level=3
# cython: initializedcheck=False
# cython: binding=True
# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def c_cov(const double[:, :]& X):
  
    cdef long points = X.shape[0]
    cdef double res
    cdef long n
    
    cdef Py_ssize_t i,j
    
    n = 0
    res = 0.
    for i in range(points):
        for j in range(i+1,points):
            d = 1. - sqrt((X[i,1]-X[j,1])**2. +
                          (X[i,2]-X[j,2])**2. +
                          (X[i,3]-X[j,3])**2.) / (X[i,4]+X[j,4])
            if d > 0:
                res += d
                n+= 1
    if n>0:
        return res/n
    else:
        return 0

def c_cov1(const double[:]& X,
           const double[:]& Y,
           const double[:]& Z,
           const double[:]& R):
  
    cdef long points = X.shape[0]
    cdef double res
    cdef long n
    
    cdef Py_ssize_t i,j
    
    n = 0
    res = 0.
    for i in range(points):
        for j in range(i+1,points):
            d = 1. - sqrt((X[i]-X[j])**2. +
                          (Y[i]-Y[j])**2. +
                          (Z[i]-Z[j])**2.) / (R[i]+R[j])
            if d > 0:
                res += d
                n+= 1
    if n>0:
        return res/n
    else:
        return 0


def _compute_all_cov(const int[:]& Aggs,
                     const double[:]& X,
                     const double[:]& Y,
                     const double[:]& Z,
                     const double[:]& R):

    cdef Py_ssize_t nsph = len(X)
    cdef Py_ssize_t nAgg = len(Aggs)

    cdef double[:] res = np.empty(len(Aggs))

    cdef Py_ssize_t isph = 0
    cdef Py_ssize_t iagg
    for iagg in range(nAgg):
        Np = Aggs[iagg]
        res[iagg] = c_cov1(X[isph:isph+Np],
                           Y[isph:isph+Np],
                           Z[isph:isph+Np],
                           R[isph:isph+Np])
        isph += Np
    return np.asarray(res)
