import cython

from cython cimport floating, integral

from cython.parallel import parallel, prange
from tqdm import tqdm

from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

import numpy as np


def eals_update(C, E, W, H, bi, regularization, num_threads=0):
    """
    """
    return _eals_update(C.indptr, C.indices,
                        C.data.astype('float32'),
                        E.data.astype('float32'),
                        W, H, bi, regularization, num_threads)


@cython.cdivision(True)
@cython.boundscheck(False)
def _eals_update(integral[:] indptr, integral[:] indices,
                 floating[:] confidence, floating[:] error,
                 floating[:, :] W, floating[:, :] H, floating[:] bi,
                 float regularization, int num_threads=0):
    """
    """
    dtype = np.float64 if floating is double else np.float32

    cdef integral N  = W.shape[0], i, j, k, index
    cdef int d = W.shape[1]
    cdef floating a, b, wik, hjk, cij, bii

    with nogil, parallel(num_threads=num_threads):
        try:
            for i in prange(N, schedule='dynamic'):
                if indptr[i] == indptr[i+1]:
                    # TODO: should we set 0s for this case similarly to implicit?
                    continue

                # update factors
                for k in range(d):
                    a = 0
                    b = 0
                    wik = W[i, k]
                    for index in range(indptr[i], indptr[i+1]):
                        j = indices[index]
                        cij = confidence[index]
                        hjk = H[j, k]
                        error[index] += wik * hjk

                        a = a + cij * hjk**2
                        b = b + cij * error[index] * hjk

                    wik = b / (a + regularization)  # new value

                    # update errors
                    for index in range(indptr[i], indptr[i+1]):
                        j = indices[index]
                        hjk = H[j, k]
                        error[index] -= wik * hjk

                    # write to the factors
                    W[i, k] = wik

                # update bias
                a = 0
                b = 0
                for index in range(indptr[i], indptr[i+1]):
                    cij = confidence[index]
                    error[index] += bi[i]
                    a = a + cij
                    b = b + cij * error[index]
                bi[i] = b / (a + regularization)  # new value

                # update errors
                for index in range(indptr[i], indptr[i+1]):
                    error[index] -= bi[i]

        finally:
            pass
            # free(x)


def compute_error(X, E, W, H, bi, bj, num_threads=0):
    """
    """
    return _compute_error(X.indptr, X.indices,
                          X.data.astype('float32'),
                          E.data.astype('float32'),
                          W, H, bi, bj, num_threads=num_threads)

@cython.cdivision(True)
@cython.boundscheck(False)
def _compute_error(integral[:] indptr, integral[:] indices,
                   floating[:] data, floating[:] error,
                   floating[:, :] W, floating[:, :] H,
                   floating[:] bi, floating[:] bj, int num_threads=0):
    """
    """
    dtype = np.float64 if floating is double else np.float32

    # refresh the error data vector
    error[:] = data.copy()

    cdef integral N  = W.shape[0], i, j, k, index
    cdef int d = W.shape[1]

    with nogil, parallel(num_threads=num_threads):
        try:
            for i in prange(N, schedule='dynamic'):
                if indptr[i] == indptr[i+1]:
                    continue

                for index in range(indptr[i], indptr[i+1]):
                    j = indices[index]
                    for k in range(d):
                        error[index] -= W[i, k] * H[j, k]
                    error[index] -= bi[i] + bj[j]
        finally:
            pass
