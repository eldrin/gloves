import os
from os.path import dirname, basename, join, splitext
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
if 'NUMBA_NUM_THREADS' not in os.environ:
    os.environ['NUMBA_NUM_THREADS'] = '4'

import logging

import numpy as np
from scipy import sparse as sp

from gloves.model import GloVe
from gloves.corpus import load_corpus


logger = logging.getLogger('Training')

RAND_STATE = os.environ.get('GLOVES_OPTIMIZE_RAND_STATE')
if RAND_STATE is not None:
    RAND_STATE = int(RAND_STATE)


def fit_model(train_data: sp.coo_matrix,
              solver: str, n_components_log2: int,
              n_iters: int, alpha: float, x_max: float,
              lr: float, l2: float,
              share_params: bool, num_threads: int, **kwargs) -> GloVe:
    """
    """
    # initiate and fit model
    d = int(2**n_components_log2)

    glove = GloVe(
        n_components=d,
        n_iters=n_iters,
        learning_rate=lr,
        l2=l2,
        alpha=alpha,
        x_max=x_max,
        solver=solver,
        dtype=np.float32,
        share_params=share_params,
        num_threads=num_threads,
        random_state=RAND_STATE
    )
    glove.fit(train_data, verbose=True)

    return glove


def fit(args):
    """
        data: cooccurrence dataset path
        out_path: path to save search result and final model
        n_components: size of latent dimensions
        solver: optimization method {'als', 'sgd'}
        n_iters: number epoch
        alpha: controls the smoothness of count measurements
        x_max: set the normalization factor (cap) for raw count measurements
        l2: L2 regularization coefficient (only for ALS)
        init: initialization weight (only for ALS)
        lr: learning rate (only for SGD)
        share_params: determine use 2 sets of embeddings (False)
                      or one (True) for words
        num_threads: control parallelization (num cores)
        quiet: set verbosity
    """
    corpus = load_corpus(args.data)
    glove = GloVe(args.n_components,
                  args.n_iters,
                  args.init,
                  args.alpha,
                  args.x_max,
                  args.solver,
                  args.l2,
                  args.learning_rate,
                  10.,  # max loss. use defaults
                  args.share_params,
                  num_threads=args.num_threads)
    glove.fit(corpus.mat, verbose=not args.quiet)

    # save the results to disk
    glove.save(join(args.path, args.out))
