import cython

from cython cimport floating, integral

from cython.parallel import parallel, prange
from tqdm import tqdm

from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset
from libc.math cimport log, sqrt, fmin

import numpy as np
from scipy import sparse as sp


cdef inline floating c_min(floating a, floating b) nogil:
    return a if a <= b else b


def eals_update(X, e_data, W, H, bi, regularization, alpha, x_max, num_threads=0):
    """
    """
    return _eals_update(X.indptr, X.indices,
                        X.data.astype('float32'), e_data,
                        W, H, bi, regularization, alpha, x_max, num_threads)


@cython.cdivision(True)
@cython.boundscheck(False)
def _eals_update(integral[:] indptr, integral[:] indices,
                 floating[:] confidence, floating[:] error,
                 floating[:, :] W, floating[:, :] H, floating[:] bi,
                 float regularization, floating alpha, floating x_max,
                 int num_threads=0):
    """
    """
    dtype = np.float64 if floating is double else np.float32

    cdef integral N  = W.shape[0], i, j, k, index, shuf_idx
    cdef int d = W.shape[1]
    cdef floating a, b, wik, hjk, cij, bii, one = 1.

    cdef integral[:] rnd_idx = np.random.permutation(N).astype('int32')

    with nogil, parallel(num_threads=num_threads):
        try:
            for shuf_idx in prange(N, schedule='dynamic'):
                # get random index
                i = rnd_idx[shuf_idx]

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


def compute_error(X, e_data, W, H, bi, bj, num_threads=0):
    """
    """
    return _compute_error(X.indptr, X.indices,
                          X.data.astype('float32'), e_data,
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

    cdef integral N  = W.shape[0], i, j, k, index
    cdef int d = W.shape[1]

    with nogil, parallel(num_threads=num_threads):
        try:
            for i in prange(N, schedule='dynamic'):
                if indptr[i] == indptr[i+1]:
                    continue

                for index in range(indptr[i], indptr[i+1]):
                    j = indices[index]
                    error[index] = log(data[index])
                    for k in range(d):
                        error[index] -= W[i, k] * H[j, k]
                    error[index] -= bi[i] + bj[j]
        finally:
            pass



def csr_tocsc(X, e_data):
    """
    """
    # new containers
    new_indptr = np.empty(X.shape[1] + 1, dtype=X.indptr.dtype)
    new_indices = np.empty(X.nnz, dtype=X.indptr.dtype)
    new_x_data = np.empty(X.nnz, dtype=X.dtype)
    new_e_data = np.empty(X.nnz, dtype=X.dtype)

    _csr_tocsc(X.shape[0], X.shape[1],
               X.indptr, X.indices, X.data, e_data,
               new_indptr, new_indices, new_x_data, new_e_data)

    A = sp.csc_matrix((new_x_data, new_indices, new_indptr), shape=X.shape)
    A.has_sorted_indices = True
    return A, new_e_data


def csc_tocsr(X, e_data):
    """
    """
    # new containers
    new_indptr = np.empty(X.shape[0] + 1, dtype=X.indptr.dtype)
    new_indices = np.empty(X.nnz, dtype=X.indptr.dtype)
    new_x_data = np.empty(X.nnz, dtype=X.dtype)
    new_e_data = np.empty(X.nnz, dtype=X.dtype)

    _csr_tocsc(X.shape[1], X.shape[0],
               X.indptr, X.indices, X.data, e_data,
               new_indptr, new_indices, new_x_data, new_e_data)

    A = sp.csr_matrix((new_x_data, new_indices, new_indptr), shape=X.shape)
    A.has_sorted_indices = True
    return A, new_e_data



@cython.cdivision(True)
@cython.boundscheck(False)
def _csr_tocsc(int n_rows, int n_cols,
               integral[:] Ap, integral[:] Aj, floating[:] Ax, floating[:] Ae,
               integral[:] Bp, integral[:] Bi, floating[:] Bx, floating[:] Be):
    """ re-implementation of scipy C++ source:
        https://github.com/scipy/scipy/blob/e4b3e6eb372b8c1d875f2adf607630a31e2a609c/scipy/sparse/sparsetools/csr.h#L418
    """
    cdef int nnz = len(Aj), col, row, jj, n, temp, last, cumsum

    # compute number of non-zero entries per column of A
    memset(&Bp[0], 0, sizeof(floating) * len(Bp))

    for n in range(nnz):
        Bp[Aj[n]] += 1
        # Bp[Aj[n]] = Bp[Aj[n]] + 1

    # cumsum the nnz per column to get Bp[]
    cumsum = 0
    for col in range(n_cols):
        temp = Bp[col]
        Bp[col] = cumsum
        cumsum = cumsum + temp
    Bp[n_cols] = nnz

    for row in range(n_rows):
        for jj in range(Ap[row], Ap[row+1]):
            col = Aj[jj]
            dest = Bp[col]

            Bi[dest] = row
            Bx[dest] = Ax[jj]
            Be[dest] = Ae[jj]

            Bp[col] += 1
            # Bp[col] = Bp[col] + 1

    last = 0
    for col in range(n_cols + 1):
        temp = Bp[col]
        Bp[col] = last
        last = temp
