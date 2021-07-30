import logging
import pickle as pkl
import numpy as np
from scipy import sparse as sp

from .solvers import ALS, SGD
from .utils import is_symmetric


logger = logging.getLogger('GloVeModel')


class GloVe(object):
    """
    """
    def __init__(self, n_components,
                 n_iters=15, alpha=3/4., x_max=100, solver='als',
                 l2=1e-3, learning_rate=0.1, max_loss=10., share_params=True,
                 use_native=True, dtype=np.float32, random_state=None,
                 num_threads=0, tokenizer=None) -> None:
        """
        """
        self.n_components = n_components
        self.l2 = l2
        self.learning_rate = learning_rate
        self.max_loss = max_loss
        self.n_iters = n_iters
        self.alpha = alpha
        self.x_max = x_max
        self.dtype = dtype
        self.solver_type = solver
        self.use_native = use_native
        self.share_params = share_params
        self.num_threads = num_threads
        self.tokenizer = tokenizer

        # set solver
        if solver == 'als':
            self.solver = ALS(n_components, l2, n_iters, alpha, x_max, use_native,
                              share_params, dtype, random_state, num_threads)
        elif solver == 'sgd':
            self.solver = SGD(n_components, learning_rate, n_iters, alpha, x_max,
                              max_loss, use_native, share_params, dtype, random_state,
                              num_threads)
        else:
            raise ValueError("[ERROR] only 'als', and 'sgd' are supported!")

    @property
    def embeddings_(self):
        """
        """
        if not hasattr(self.solver, 'embeddings_'):
            raise Exception('[ERROR] model should be fitted first!')
        return self.solver.embeddings_

    def fit(self, X, verbose=False):
        """
        """
        self.solver.fit(X, verbose)

    def score(self, X, weighted=False):
        """
        """
        return self.solver.score(X, weighted)

    def most_similar(self, word, topn=5):
        """
        """
        raise NotImplementedError()

    def __getitem__(self, word):
        """ outputs the word vector if the word exists
        """
        raise NotImplementedError()

    def save(self, out_fn):
        """
        """
        configs = {
            'n_components': self.n_components,
            'l2': self.l2,
            'learning_rate': self.learning_rate,
            'max_loss': self.max_loss,
            'n_iters': self.n_iters,
            'alpha': self.alpha,
            'x_max': self.x_max,
            'dtype': self.dtype,
            'solver_type': self.solver_type,
            'user_native': self.use_native,
            'share_params': self.share_params,
            'num_threads': self.num_threads
        }
        params = {
            'W': self.solver.embeddings_['W'],
            'bi': self.solver.embeddings_['bi']
        }
        if not self.share_params:
            params.update({
                'H': self.solver.embeddings_['H'],
                'bj': self.solver.embeddings_['bj']
            })

        with open(out_fn, 'rb') as fp:
            pkl.dump({
                'configs': configs,
                'params': params
            }, fp)

    @classmethod
    def from_file(cls, fn):
        """
        """
        with open(fn, 'rb') as fp:
            saved = pkl.load(fp)
            new_glove = cls(**saved['config'])
        new_glove.solver.embeddings_ = saved['params']
        return new_glove
