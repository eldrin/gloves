import numpy as np
from scipy import sparse as sp

from tqdm import tqdm
from . import _ials
from .base import SolverBase


UNSEARCHABLES = {'lr', 'max_loss', 'x_max', 'alpha'}


class IALS(SolverBase):
    def __init__(self, n_components, l2=1e-3, n_iters=15,
                 beta=10., eps=1., use_native=True, share_params=True,
                 dtype=np.float32, random_state=None, num_threads=0):
        """
        """
        super().__init__(n_components, l2, n_iters, 3/4., 100., use_native,
                         share_params, dtype, random_state, num_threads)
        self.beta = beta
        self.eps = eps

    def fit(self, X, verbose=True):
        """
        """
        # force to convert
        if not sp.isspmatrix_csr(X):
            X = X.tocsr(copy=False)
        X.data = 1. + self.beta * np.log((X.data / self.eps) + 1.)

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

        # from now on, it's containing the confidence
        with tqdm(total=self.n_iters, ncols=80, disable=not verbose) as prog:
            for n in range(self.n_iters):

                self.solver(X, self.embeddings_['W'], self.embeddings_['H'].copy(),
                            self.l2, num_threads=self.num_threads)
                X = X.T.tocsr()

                self.solver(X, self.embeddings_['H'], self.embeddings_['W'].copy(),
                            self.l2, num_threads=self.num_threads)
                X = X.T.tocsr()

                if self._is_unhealthy():
                    print('[ERROR] Training failed! nan or inf found')
                    break
                prog.update()

    @property
    def solver(self):
        # TODO: add python solvers
        # return _als.eals_update if self.use_native else eals_update
        return _ials.least_squares_cg


