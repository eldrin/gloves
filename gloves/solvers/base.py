import logging
import numpy as np
from scipy import sparse as sp

from . import _als
from ..utils import is_symmetric


logger = logging.getLogger('SolverBase')


class SolverBase(object):
    """
    """

    def __init__(self, n_components, l2=1e-3, n_iters=15, alpha=3/4., x_max=100,
                 use_native=True, share_params=True, dtype=np.float32,
                 random_state=None, num_threads=0) -> None:
        """
        """
        self.n_components = n_components
        self.l2 = l2
        self.n_iters = n_iters
        self.alpha = alpha
        self.x_max = x_max
        self.dtype = dtype
        self.use_native = use_native
        self.share_params = share_params
        self.num_threads = num_threads

        if isinstance(random_state, int) or random_state is None:
            self.rng = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            raise ValueError('[ERROR] `random_state` should either be seed'
                             ' {None, int} or an instance of np.random.RandomState')

    def _init_params(self, n: int) -> None:
        """
        """
        # init vectors
        W = self.rng.rand(n, self.n_components).astype(self.dtype)
        W = (W - .5) / self.n_components

        # init bias
        b = np.zeros((n,), dtype=self.dtype)

        return W, b

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
        self.compute_error(X_, E_,
                           self.embeddings_['W'], self.embeddings_['H'],
                           self.embeddings_['bi'], self.embeddings_['bj'])

        if weighted:
            # weighted mean squared error
            return np.mean(C_.data * (E_.data)**2)
        else:
            # uniform mse
            return np.mean((E_.data)**2)

    def _is_unhealthy(self) -> bool:
        """
        """
        if not hasattr(self, 'embeddings_'):
            raise Exception('[ERROR] GloVeALS should be fitted first!')

        # check nan
        is_nan = np.any([np.any(np.isnan(param))
                         for param in self.embeddings_.values()])

        # check inf
        is_inf = np.any([np.any(np.isnan(param))
                         for param in self.embeddings_.values()])

        return any([is_nan, is_inf])

    @property
    def compute_error(self):
        return _als.compute_error if self.use_native else compute_error

    @property
    def solver(self):
        raise NotImplementedError()

    def fit(self, X, verbose=True, compute_loss=False):
        raise NotImplementedError()


def transform(X, x_max=100, alpha=3/4., dtype=np.float32):
    """
    """
    # transform target values
    X_ = X.copy()
    X_.data = np.log(X_.data).astype(dtype)

    # prepare confidence function
    C_ = X.copy().astype(dtype)
    C_.data = np.minimum(1., X.data / x_max) ** alpha

    return X_, C_


def compute_error(X, E, W, H, bi, bj, *args, **kwargs):
    """
    """
    return _compute_error(X.data, E.data, X.indices, X.indptr, W, H, bi, bj)


def _compute_error(data, error, indices, indptr, W, H, bi, bj):
    """
    """
    N, d = W.shape
    rnd_idx = np.random.permutation(N)
    for n in range(N):
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
