from typing import Optional
import os
from os.path import join
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
if 'NUMBA_NUM_THREADS' not in os.environ:
    os.environ['NUMBA_NUM_THREADS'] = '4'

import logging

import numpy as np
from scipy import sparse as sp

from gloves.model import GloVe
from gloves.corpus import load_corpus
from gloves.utils import init_tokenizer

from tokenizers import Tokenizer


logger = logging.getLogger('Training')

RAND_STATE = os.environ.get('GLOVES_OPTIMIZE_RAND_STATE')
if RAND_STATE is not None:
    RAND_STATE = int(RAND_STATE)


def fit_model(train_data: sp.coo_matrix,
              solver: str, n_components_log2: int,
              n_iters: int, alpha: float, x_max: float, beta: float, eps: float,
              share_params: bool, num_threads: int, tokenizer: Tokenizer,
              lr: Optional[float]=None, l2: Optional[float]=None,
              **kwargs) -> GloVe:
    """
    """
    # initiate and fit model
    d = int(2**n_components_log2)

    glove = GloVe(
        n_components  = d,
        n_iters       = n_iters,
        learning_rate = lr,
        l2            = l2,
        alpha         = alpha,
        x_max         = x_max,
        beta          = beta,
        eps           = eps,
        solver        = solver,
        dtype         = np.float32,
        share_params  = share_params,
        num_threads   = num_threads,
        random_state  = RAND_STATE,
        tokenizer     = tokenizer
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
    corpus = load_corpus(args.corpus)
    glove = GloVe(n_components  = args.n_components,
                  n_iters       = args.n_iters,
                  alpha         = args.alpha,
                  x_max         = args.x_max,
                  beta          = args.beta,
                  eps           = args.eps,
                  solver        = args.solver,
                  l2            = args.l2,
                  learning_rate = args.learning_rate,
                  max_loss      = 10.,  # max loss. use defaults
                  share_params  = args.share_params,
                  num_threads   = args.num_threads,
                  tokenizer     = corpus._tokenizer)
    glove.fit(corpus.mat, verbose=not args.quiet)

    # save the results to disk
    glove.save(join(args.path, args.out))
