import numpy as np
from scipy import sparse as sp

from tqdm import tqdm
from . import _sgd
from .base import SolverBase


class SGD(SolverBase):
    def __init__(self, n_components, learning_rate=0.1, n_iters=15,
                 alpha=3/4., x_max=100, max_loss=10., use_native=True,
                 share_params=True, dtype=np.float32, random_state=None,
                 num_threads=0):
        """

        this class implement own version of GloVe based on
        [python-glove](https://github.com/maciejkula/glove-python)

        it learns the embeddings using SGD (ADAGRAD) solver.
        it misses L2 regularization, which could be added.

        """
        super().__init__(n_components, 0, n_iters, alpha, x_max, use_native,
                         share_params, dtype, random_state, num_threads)

        self.learning_rate = learning_rate
        self.max_loss = max_loss  # gradient clipper

    def fit(self, X, verbose=True):
        """
        """
        # force to convert
        if not sp.isspmatrix_coo(X):
            X = X.tocoo()

        # initialize parameters
        N = X.shape[0]
        d = self.n_components

        W, bi = self._init_params(N)

        # tracking cumulative gradient for ADAGRAD updates
        dW, dbi = self._init_grads(W, bi)
        self.embeddings_ = dict(W=W, bi=bi, dW=dW, dbi=dbi)

        if self.share_params:
            self.embeddings_.update(dict(H=W, bj=bi, dH=dW, dbj=dbi))
        else:
            H, bj = self._init_params(N)
            dH, dbj = self._init_grads(H, bj)
            self.embeddings_.update(dict(H=H, bj=bj, dH=dH, dbj=dbj))

        # compute error matrix
        with tqdm(total=self.n_iters, ncols=80, disable=not verbose) as prog:
            for _ in range(self.n_iters):
                self.solver(X,
                            self.embeddings_['W'], self.embeddings_['dW'],
                            self.embeddings_['H'], self.embeddings_['dH'],
                            self.embeddings_['bi'], self.embeddings_['dbi'],
                            self.embeddings_['bj'], self.embeddings_['dbj'],
                            self.learning_rate, self.alpha, self.x_max,
                            self.max_loss, num_threads=self.num_threads)

                if self._is_unhealthy():
                    print('[ERROR] Training failed! nan or inf found')
                    break

                prog.update()

    @staticmethod
    def _init_grads(*params):
        """
        """
        return tuple(map(np.ones_like, params))

    @property
    def solver(self):
        return _sgd.sgd_update if self.use_native else sgd_update


def sgd_update(X, W, dW, H, dH, bi, dbi, bj, dbj,
               learn_rate, alpha, x_max, max_loss, *args, **kwargs):
    """ no regularization?
    """
    nnz = X.nnz
    rnd_idx = np.random.permutation(nnz)
    total_error = 0
    for n in rnd_idx:
        # parse triplet
        i, j, x = X.row[n], X.col[n], X.data[n]

        # compute prediction / error
        pred = W[i].T @ H[j] + bi[i] + b[j]
        conf = min(1., (x / x_max)) ** alpha
        err = pred - np.log(x)
        total_error += conf * (err ** 2) / nnz
        loss = conf * err

        # clip the loss for numerical stability
        loss = min(max(loss, -max_loss), max_loss)

        # update factors
        _update_factor(i, j, loss, W, H, dW, learn_rate)
        _update_factor(j, i, loss, H, W, dH, learn_rate)

        # update biases
        _update_bias(i, j, loss, bi, bj, dbi, learn_rate)
        _update_bias(j, i, loss, bj, bi, dbj, learn_rate)


def _update_factor_adagrad(word_i, word_j, loss,
                           word_vec_a, word_vec_b, sum_grad_a,
                           learn_rate, eps=1e-6):
    """
    """
    cur_lr = learn_rate / np.sqrt(sum_grad_a[word_i] + eps)
    grad = loss * word_vec_b[word_j]
    word_vec_a[word_i] -= cur_lr * grad
    sum_grad_a[word_i] += grad ** 2


def _update_bias_adagrad(word_i, word_j, loss,
                         word_bias_a, word_bias_b, sum_grad_a,
                         learn_rate, eps=1e-6):
    """
    """
    cur_lr = learn_rate / np.sqrt(sum_grad_a[word_i] + eps)
    word_bias_a[word_i] -= cur_lr * loss
    sum_grad_a[word_i] += loss ** 2
