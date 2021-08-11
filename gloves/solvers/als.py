import numpy as np
from scipy import sparse as sp

from tqdm import tqdm
from . import _als
from .base import SolverBase


UNSEARCHABLES = {'lr', 'max_loss'}


class ALS(SolverBase):
    def __init__(self, n_components, l2=1e-3, n_iters=15,
                 alpha=3/4., x_max=100, use_native=True, share_params=True,
                 dtype=np.float32, random_state=None, num_threads=0):
        """
        """
        super().__init__(n_components, l2, n_iters, alpha, x_max, use_native,
                         share_params, dtype, random_state, num_threads)

    def fit(self, X, verbose=True):
        """
        """
        # force to convert
        if not sp.isspmatrix_csr(X):
            X = X.tocsr(copy=False)

        # initialize parameters
        N = X.shape[0]
        d = self.n_components

        W, bi = self._init_params(N)
        self.embeddings_ = dict(W=W, bi=bi)
        if self.share_params:
            self.embeddings_.update(dict(H=W, bj=bi))
        else:
            H, bj = self._init_params(N)
            self.embeddings_.update(dict(H=H, bj=bj))

        # compute error matrix
        # TODO: we may be able to save lots of memory by only copying "data" field
        #       as coordinate infor is identical to "X/C"
        e_data = X.data.copy().astype('float32')
        self.compute_error(X, e_data,
                           self.embeddings_['W'], self.embeddings_['H'],
                           self.embeddings_['bi'], self.embeddings_['bj'],
                           num_threads=self.num_threads)

        # from now on, it's containing the confidence
        X.data = np.minimum(1., X.data / self.x_max) ** self.alpha
        with tqdm(total=self.n_iters, ncols=80, disable=not verbose) as prog:
            for n in range(self.n_iters):

                self.solver(X, e_data,
                            self.embeddings_['W'],
                            self.embeddings_['H'].copy(),
                            self.embeddings_['bi'],
                            self.l2, self.alpha, self.x_max,
                            num_threads=self.num_threads)
                X, e_data = _als.csr_tocsc(X, e_data)

                self.solver(X, e_data,
                            self.embeddings_['H'],
                            self.embeddings_['W'].copy(),
                            self.embeddings_['bj'],
                            self.l2, self.alpha, self.x_max,
                            num_threads=self.num_threads)
                X, e_data = _als.csc_tocsr(X, e_data)

                if self._is_unhealthy():
                    print('[ERROR] Training failed! nan or inf found')
                    break
                prog.update()

    @property
    def solver(self):
        return _als.eals_update if self.use_native else eals_update


def _partial_update_factor(i, conf, err, ind, W, H, bi, lmbda):
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


def _partial_update_bias(i, conf, err, ind, W, H, bi, lmbda):
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


def eals_update(C, E, W, H, bi, regularization, *args, **kwargs):
    return _update_factor(C.data, E.data, C.indices, C.indptr,
                          W, H, bi, regularization)


def _update_factor(confidence, error, indices, indptr, W, H, bi, lmbda):
    """
    """
    N, d = W.shape
    rnd_idx = np.random.permutation(N)
    for n in range(N):
        i = rnd_idx[n]
        i0, i1 = indptr[i], indptr[i + 1]
        if i1 == i0:
            continue

        ind = indices[i0:i1]
        conf = confidence[i0:i1]
        err = error[i0:i1]

        _partial_update_factor(i, conf, err, ind, W, H, bi, lmbda)
        _partial_update_bias(i, conf, err, ind, W, H, bi, lmbda)
