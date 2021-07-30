import cython

from cython cimport floating, integral

from cython.parallel import parallel, prange
from tqdm import tqdm

from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from libc.math cimport log, sqrt

import numpy as np


cdef inline floating c_min(floating a, floating b) nogil:
    return a if a <= b else b

cdef inline floating c_max(floating a, floating b) nogil:
    return a if a > b else b


def sgd_update(X, W, dW, H, dH, bi, dbi, bj, dbj,
               learn_rate, alpha, x_max, max_loss, num_threads=0):
    """
    """
    return _sgd_update(
        X.row, X.col, X.data.astype('float32'),
        W, dW, H, dH, bi, dbi, bj, dbj,
        learn_rate, alpha, x_max, max_loss,
        num_threads = num_threads
    )


@cython.cdivision(True)
@cython.boundscheck(False)
def _sgd_update(integral[:] row, integral[:] col, floating[:] count,
                floating[:, :] W, floating[:, :] dW,
                floating[:, :] H, floating[:, :] dH,
                floating[:] bi, floating[:] dbi,
                floating[:] bj, floating[:] dbj,
                float learn_rate, float alpha, float x_max, float max_loss,
                floating eps=1e-6, int num_threads=0):
    """
    """
    dtype = np.float64 if floating is double else np.float32

    cdef integral nnz = count.shape[0], d = W.shape[1]
    cdef integral n, i, j, k, shuf_idx
    cdef floating x, pred, conf, err, loss, grad, cur_lr

    cdef integral[:] rnd_idx = np.random.permutation(nnz).astype('int32')

    with nogil, parallel(num_threads=num_threads):
        try:
            for shuf_idx in prange(nnz, schedule='dynamic'):
                # parse triplet
                n = rnd_idx[shuf_idx]
                i = row[n]
                j = col[n]
                x = count[n]

                # compute prediction / error
                pred = 0.
                for k in range(d):
                    pred = pred + W[i, k] * H[j, k]
                pred = pred + bi[i] + bj[j]

                conf = c_min(1., (x / x_max)) ** alpha
                err = pred - log(x)
                loss = conf * err

                # clip the loss for numerical stability
                loss = c_min(c_max(loss, -max_loss), max_loss)

                # update factors
                for k in range(d):
                    cur_lr = learn_rate / sqrt(dW[i, k] + eps)
                    grad = loss * H[j, k]
                    W[i, k] = W[i, k] - cur_lr * grad
                    dW[i, k] += grad ** 2

                    cur_lr = learn_rate / sqrt(dH[j, k] + eps)
                    grad = loss * W[i, k]
                    H[j, k] = H[j, k] - cur_lr * grad
                    dH[i, k] += grad ** 2

                # update biases
                cur_lr = learn_rate / sqrt(dbi[i] + eps)
                bi[i] -= cur_lr * loss
                dbi[i] += loss ** 2

                cur_lr = learn_rate / sqrt(dbj[j] + eps)
                bj[j] -= cur_lr * loss
                dbj[j] += loss ** 2

        finally:
            pass
