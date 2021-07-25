import numpy as np
from scipy import sparse as sp
import numba as nb

from tqdm import tqdm


class GloVeALS:
    def __init__(self, n_components, l2=1e-3, init=1e-3, n_iters=15,
                 alpha=3/4., x_max=100, dtype=np.float32):
        """
        """
        self.n_components = n_components
        self.l2 = l2
        self.init = init
        self.n_iters = n_iters
        self.alpha = alpha
        self.x_max = x_max
        self.dtype = dtype

    def fit(self, X, verbose=True, compute_loss=False):
        """
        """
        # force to convert
        if not sp.isspmatrix_csr(X):
            X = X.tocsr()

        # transform input data
        X_, C_ = transform(X, self.x_max, self.alpha, self.dtype)

        # initialize parameters
        N = X.shape[0]
        d = self.n_components

        W = np.random.randn(N, d).astype(self.dtype) * self.init
        H = np.random.randn(N, d).astype(self.dtype) * self.init
        bi = np.zeros((N,), dtype=self.dtype)
        bj = np.zeros((N,), dtype=self.dtype)

        # compute error matrix
        E_ = X_.copy()
        compute_error(X_.data, E_.data, X_.indices, X_.indptr,
                      W, H, bi, bj)

        Xt_ = X_.T.tocsr()
        Ct_ = C_.T.tocsr()
        Et_ = E_.T.tocsr()
        if compute_loss:
            self.losses = [np.mean(C_.data * E_.data**2)]
        healthy = True
        with tqdm(total=self.n_iters, ncols=80, disable=not verbose) as prog:
            for n in range(self.n_iters):
                E_ = Et_.T.tocsr()
                update_factor(
                    C_.data, E_.data, X_.indices, X_.indptr,
                    W, H, bi, bj, self.l2
                )
                Et_ = E_.T.tocsr()
                update_factor(
                    Ct_.data, Et_.data, Xt_.indices, Xt_.indptr,
                    H, W, bj, bi, self.l2
                )
                if self._is_unhealthy(W, H, bi, bj):
                    healthy = False
                    print('[ERROR] Training failed! nan or inf found')
                    break

                if compute_loss:
                    self.losses.append(np.mean(C_.data * E_.data**2))
                prog.update()

        if healthy:
            self.embeddings_ = {
                'W': W, 'H': H, 'bi': bi, 'bj': bj
            }

    def score(self, X, weighted=True):
        """
        """
        if not hasattr(self, 'embeddings_'):
            raise Exception('[ERROR] GloVeALS should be fitted first!')

        # force to convert
        if not sp.isspmatrix_csr(X):
            X = X.tocsr()

        # transform input data
        X_, C_ = transform(X, self.x_max, self.alpha, self.dtype)

        # compute errors
        E_ = X_.copy()
        compute_error(X_.data, E_.data, X_.indices, X_.indptr,
                      self.embeddings_['W'], self.embeddings_['H'],
                      self.embeddings_['bi'], self.embeddings_['bj'])

        if weighted:
            # weighted mean squared error
            return np.mean(C_.data * (E_.data)**2)
        else:
            # uniform mse
            return np.mean((E_.data)**2)

    @staticmethod
    def _is_unhealthy(*params):
        """
        """
        # check nan
        is_nan = np.any([np.any(np.isnan(param)) for param in params])

        # check inf
        is_inf = np.any([np.any(np.isnan(param)) for param in params])

        return any([is_nan, is_inf])


@nb.njit(
    [
        "void(i8, f4[:], f4[:], i4[:], f4[:,::1], f4[:,::1], f4[::1], f4[::1], f8)",
        "void(i8, f8[:], f8[:], i4[:], f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8)"
    ],
    cache=True
)
def _partial_update_factor(i, conf, err, ind, W, H, bi, bj, lmbda):
    """
    """
    for k in range(W.shape[1]):
        a = W.dtype.type(0.)
        b = W.dtype.type(0.)
        wik = W[i, k]
        for m in range(len(ind)):
            j = ind[m]
            hjk = H[j, k]
            cij = conf[m]

            # get temporary error
            err[m] += wik * hjk

            a += cij * hjk**2
            b += cij * err[m] * hjk

        wik = b / (a + lmbda)  # new value

        # update errors
        for m in range(len(ind)):
            j = ind[m]
            hjk = H[j, k]
            err[m] -= wik * hjk

        W[i, k] = wik


@nb.njit
def __solve_cg(A, b, x0, n_iters=3, eps=1e-20):
    """
    """
    d = len(b)
    r = b - A @ x0
    p = r.copy()
    rsold = np.sum(r**2)
    if rsold**.5 < eps:
        return x0

    for it in range(n_iters):
        Ap = A @ p.T
        pAp = p @ Ap
        alpha = rsold / pAp
        x0 += alpha * p
        r -= alpha * Ap

        rsnew = np.sum(r**2)
        if rsnew**.5 < eps:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x0


@nb.njit(
    [
        "void(i8, f4[:], f4[:], i4[:], f4[:,::1], f4[:,::1], f4[::1], f4[::1], f8)",
        "void(i8, f8[:], f8[:], i4[:], f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8)"
    ],
    cache=True
)
def _partial_update_factor_cg(i, conf, err, ind, W, H, bi, bj, lmbda):
    """
    """
    d = H.shape[-1]
    dtype = W.dtype
    l = dtype.type(lmbda)

    w = W[i].copy()  # x0
    h = np.ascontiguousarray(H[ind])

    # compute b
    err[:] += w @ h.T  # temporarily update err
    b = (err * conf) @ h

    # compute A
    CH = np.expand_dims(conf, axis=-1) * h
    A = h.T @ CH + l * np.eye(d, dtype=dtype)

    # solve it
    w = __solve_cg(A, b, w)

    # update errors
    err[:] -= w @ h.T

    # update W
    W[i] = w


@nb.njit(
    [
        "void(i8, f4[:], f4[:], i4[:], f4[:,::1], f4[:,::1], f4[::1], f4[::1], f8)",
        "void(i8, f8[:], f8[:], i4[:], f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8)",
    ],
    cache=True
)
def _partial_update_bias(i, conf, err, ind, W, H, bi, bj, lmbda):
    """
    """
    a = W.dtype.type(0.)
    b = W.dtype.type(0.)

    bii = bi[i]
    for m in range(len(ind)):
        cij = conf[m]

        # get temporary error
        err[m] += bii

        a += cij
        b += cij * err[m]

    bii = b / (a + lmbda)  # new value

    # update errors
    for m in range(len(ind)):
        err[m] -= bii

    bi[i] = bii


@nb.njit(
    [
        "void(f4[::1], f4[::1], i4[::1], i4[::1], f4[:,::1], f4[:,::1], f4[::1], f4[::1], f8)",
        "void(f8[::1], f8[::1], i4[::1], i4[::1], f8[:,::1], f8[:,::1], f8[::1], f8[::1], f8)"
    ],
    parallel=True,
    nogil=True,
    cache=True
)
def update_factor(confidence, error, indices, indptr, W, H, bi, bj, lmbda):
    """
    """
    N, d = W.shape
    rnd_idx = np.random.permutation(N)
    for n in nb.prange(N):
        i = rnd_idx[n]
        i0, i1 = indptr[i], indptr[i + 1]
        if i1 == i0:
            continue

        ind = indices[i0:i1]
        conf = confidence[i0:i1]
        err = error[i0:i1]

        _partial_update_factor(i, conf, err, ind, W, H, bi, bj, lmbda)
        # _partial_update_factor_cg(i, conf, err, ind, W, H, bi, bj, lmbda)
        _partial_update_bias(i, conf, err, ind, W, H, bi, bj, lmbda)


@nb.njit(
    [
        "void(f4[::1], f4[::1], i4[::1], i4[::1], f4[:,::1], f4[:,::1], f4[::1], f4[::1])",
        "void(f8[::1], f8[::1], i4[::1], i4[::1], f8[:,::1], f8[:,::1], f8[::1], f8[::1])"
    ],
    parallel=True,
    nogil=True,
    cache=True
)
def compute_error(data, error, indices, indptr, W, H, bi, bj):
    """
    """
    N, d = W.shape
    rnd_idx = np.random.permutation(N)
    for n in nb.prange(N):
        i = rnd_idx[n]
        i0, i1 = indptr[i], indptr[i + 1]
        if i1 == i0:
            continue

        for m in range(i0, i1):
            j = indices[m]
            error[m] = data[m]  # yij
            for k in range(W.shape[1]):
                error[m] -= W[i, k] * H[j, k]
            error[m] -= bi[i] + bj[j]


def transform(X, x_max=100, alpha=3/4., dtype=np.float32):
    """
    """
    # transform target values
    X_ = X.copy()
    X_.data = np.log(X_.data + 1.).astype(dtype)

    # prepare confidence function
    C_ = X.copy().astype(dtype)
    lt_xmax = X.data < x_max
    C_.data[lt_xmax]  = (X.data[lt_xmax] / x_max)**alpha
    C_.data[~lt_xmax] = 1.

    return X_, C_
