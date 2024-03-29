from typing import Optional
import logging
import pickle as pkl
import numpy as np
from scipy.spatial.distance import cdist

from .solvers import ALS, SGD, IALS
from .utils import argpart_sort, init_tokenizer


logger = logging.getLogger('GloVeModel')


class GloVe(object):
    """
    """
    def __init__(self, n_components,
                 n_iters=15, alpha=3/4., x_max=100, beta=1e+1, eps=1e-0,
                 solver='als', l2=1e-3, learning_rate=0.1, max_loss=10.,
                 share_params=True, use_native=True, dtype=np.float32,
                 random_state=None, num_threads=0, tokenizer=None) -> None:
        """
        """
        self.n_components = n_components
        self.l2 = l2
        self.learning_rate = learning_rate
        self.max_loss = max_loss
        self.n_iters = n_iters
        self.alpha = alpha
        self.x_max = x_max
        self.beta = beta
        self.eps = eps
        self.dtype = dtype
        self.solver_type = solver
        self.use_native = use_native
        self.share_params = share_params
        self.num_threads = num_threads

        if tokenizer is None:
            logger.warning('[Warning] tokenizer is not given. '
                           'intializing default tokenizer...')
            self._tokenizer = init_tokenizer()
            self._using_default_tokenizer = True
        else:
            self._tokenizer = tokenizer
            self._using_default_tokenizer = False

        # set solver
        if solver == 'als':
            self.solver = ALS(n_components, l2, n_iters, alpha, x_max,
                              use_native, share_params, dtype, random_state,
                              num_threads)
        elif solver == 'sgd':
            self.solver = SGD(n_components, learning_rate, n_iters, alpha, x_max,
                              max_loss, use_native, share_params, dtype,
                              random_state, num_threads)
        elif solver == 'ials':
            self.solver = IALS(n_components, l2, n_iters, beta, eps,
                               use_native, share_params, dtype, random_state,
                               num_threads)
        else:
            raise ValueError("[ERROR] only 'als', 'sgd', and 'ials' are supported!")

    def set_tokenizer(self, tokenizer):
        """ set internal tokenizer

        in case it's needed to overridden or updated.
        """
        self._tokenizer = tokenizer

    @property
    def is_unhealthy(self):
        return self.solver._is_unhealthy()

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

    def most_similar(self, word, topn=5, method='cosine'):
        """ Find most close words based on the cosine-distance

        Currently it finds the exact neighbors, which means it
        can get exponentially slow once the word vector gets larger.
        (both in vocabulary size and the dimensionality)

        Possible solutions would be:
            1) pre-building neighbors in model file -> makes the model dump larger
            2) falls back to approximated nearest neighbors (ANN)
               once performance is concern -> needs ANN algorithms

        TODO: this should be optimized
        """
        word_id = self.get_id(word)
        if word_id is None:
            raise ValueError(f'[ERROR] {word} is not found in the dictionary!')

        # aliasing.
        emb = self.solver.embeddings_['W']
        word_vec = emb[word_id]

        # not normalized by the query word vector, but irrelevant for computing ranking
        if method == 'cosine':
            score = 1 - cdist(word_vec[None], emb, 'cosine')[0]
        elif method == 'score':
            score = word_vec @ emb.T
        else:
            raise ValueError('[ERROR] only `cosine` and `score` are supported!')

        neighbors = argpart_sort(score, topn, ascending=False)

        return [
            (self._tokenizer.decode([neighbor]), score[neighbor])
            for neighbor in neighbors
        ]

    def get_id(self, word):
        """ outputs the word vector if the word exists
        """
        if not hasattr(self, '_tokenizer') or self._tokenizer is None:
            raise ValueError('[ERROR] tokenizer is not set! set it first '
                             'using `set_tokenizer`')
        tok = self._tokenizer.encode(word)
        if len(tok.ids) > 1:
            return None
        else:
            return tok.ids[0]

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        """
        index = self.get_id(word)
        if index is not None:
            return self.solver.embeddings_['W'][index]
        else:
            return None

    def encode(self, word_or_sentence: str) -> tuple[np.ndarray, list[int]]:
        """
        TODO: check whether the output is too weird or not
              (i.e., by checking every tokens are trivial such as alphabets)
        """
        tok = self._tokenizer.encode(word_or_sentence)
        ids = tok.ids
        return self.solver.embeddings_['W'][ids], ids

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
            'beta': self.beta,
            'eps': self.eps,
            'dtype': self.dtype,
            'solver': self.solver_type,
            'use_native': self.use_native,
            'share_params': self.share_params,
            'num_threads': self.num_threads,
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

        with open(out_fn, 'wb') as fp:
            pkl.dump({
                'configs': configs,
                'params': params,
                'tokenizer': self._tokenizer.to_str()
            }, fp)

    @classmethod
    def from_file(cls, fn):
        """
        """
        with open(fn, 'rb') as fp:
            saved = pkl.load(fp)
            tokenizer = init_tokenizer(dump_str=saved['tokenizer'])
            new_glove = cls(tokenizer=tokenizer, **saved['configs'])
        new_glove.solver.embeddings_ = saved['params']
        return new_glove
