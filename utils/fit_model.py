import os
from os.path import dirname, basename, join, splitext
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
if 'NUMBA_NUM_THREADS' not in os.environ:
    os.environ['NUMBA_NUM_THREADS'] = '4'

from typing import Optional, TypedDict, Union

import json
import glob
import argparse
from functools import partial
import pickle as pkl

import numpy as np
from scipy import sparse as sp
from scipy.stats import kendalltau

import skopt
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

from tokenizers import Tokenizer

from gloves.model import GloVe
from gloves.evaluation import split_data
from gloves.corpus import Corpus, load_corpus


# ============== Data Object Definitions =================

class GloVeDataSpec(TypedDict):
    window_size: int

class LyricsDataSpec(GloVeDataSpec):
    testing_fold: int

# total=False for tokenizer
# TODO: we'll have `Required` as of python 3.10
class GloVeData(TypedDict, total=False):
    X: sp.coo_matrix
    tokens_inv_map: dict[str, int]
    tokenizer: Optional[str]
    spec: GloVeDataSpec

# alias for the score object
class EvaluationScore(TypedDict):
    tau: float
    p: float
    nan_rate:float

# alias for the other data objects
FaruquiEvalSet = dict[str, dict[frozenset, float]]
EvaluationSet = Union[FaruquiEvalSet, sp.coo_matrix]
EvaluationResult = dict[str, EvaluationScore]
Predictions = dict[str, tuple[str, str]]

# aliasing parameter tuple. (note: see the SPACE variable below)
# HyperParams = tuple[int, int, int, float, float, bool, float]
HyperParams = tuple[int, int, int, float, float, float, bool, float]  # with `init`

# ============== Data Object Definitions =================

NUM_THREADS = int(os.environ['NUMBA_NUM_THREADS'])
RAND_STATE = 1234
# np.random.seed(RAND_STATE)

EVAL_DATA_PATH = join(
    dirname(__file__),
    '../eval-word-vectors/data/word-sim/'
)

# setup the search space
SPACE = [
    Integer(2, 7, name='window_size_factor2'),
    Integer(4, 9, name='n_components_log2'),
    Integer(10, 120, name='n_iters'),
    Real(0.5, 1, name='alpha'),
    Real(1e+1, 1e+2, 'log-uniform', name='x_max'),
    Real(1e-3, 1e+1, 'log-uniform', name='init'),
    # Categorical([True, False], name='share_params')
    Categorical([True], name='share_params')  # for checking share params woring
]


def load_eval_dataset(path: str) -> FaruquiEvalSet:
    """
    """
    valid_fns = glob.glob(join(path, '*.txt'))
    if len(valid_fns) == 0:
        raise ValueError('[ERROR] no validation dataset found!')

    eval_set = {}
    for fn in valid_fns:
        dataset = basename(fn).replace('.txt', '')
        eval_set[dataset] = {}
        sep = None
        with open(fn) as f:
            for line in f:
                if sep is None:
                    if '\t' in line:
                        sep = '\t'
                    elif ' ' in line:
                        sep = ' '

                w1, w2, sim = line.replace('\n', '').split(sep)

                key = frozenset((w1, w2))
                eval_set[dataset][key] = float(sim)

    return eval_set


def check_exists(token: str,
                 token_inv_map: dict[str, int],
                 tokenizer: Optional[Tokenizer] = None) -> Optional[int]:
    """
    """
    # here we check if the whole form of token is in tokenizer
    # otherwise, it'll give split list of sub-tokens
    if tokenizer is not None:
        tok = tokenizer.encode(token)
        if len(tok.tokens) == 1:
            return tok.ids[0]
        else:
            return None

    # if tokenizer is not given, use the token inverse map
    if token in token_inv_map:
        return token_inv_map[token]

    # otherwise, return None, as indication of `not-found`
    return None


def compute_similarities(glove: GloVe,
                         corpus: Corpus,
                         eval_set: EvaluationSet,
                         token_inv_map: dict[str, int]) -> Predictions:
    """
    """
    eps = 1e-20

    W = glove.embeddings_['W']

    # normalized embeddings for computing cosine distance easily
    W = W / (np.linalg.norm(W, axis=1)[:, None] + eps)

    predictions = {}
    for dataset, ratings in eval_set.items():
        predictions[dataset] = {}

        for pair, rating in ratings.items():
            if len(pair) == 1:
                w1 = w2 = next(iter(pair))
            else:
                w1, w2 = pair

            # can't estimate the similarity due to the coverage
            i1 = check_exists(w1, token_inv_map, corpus._tokenizer)
            i2 = check_exists(w2, token_inv_map, corpus._tokenizer)
            if i1 is None or i2 is None:
                predictions[dataset][pair] = None
                continue

            predictions[dataset][pair] = W[i1] @ W[i2]

    return predictions


def compute_scores(eval_set: FaruquiEvalSet,
                   predictions: Predictions) -> EvaluationResult:
    """
    """
    scores = {}
    for dataset, pred in predictions.items():
        judges = eval_set[dataset]

        # get the prediction / jugement pairs
        pred_judge = [(judges[k], v) for k, v in pred.items()]
        p, j = list(zip(*[x for x in pred_judge if x[1] is not None]))

        # compute missing rate
        n_nans = len(pred) - len(j)
        nan_rate = n_nans / len(pred)

        # compute (non-parametric) correlation
        tau = kendalltau(p, j)

        # register the row
        scores[dataset] = {'tau': tau.correlation,
                           'p': tau.pvalue,
                           'nan_rate': nan_rate}
    return scores


def is_model_bad(glove: GloVe) -> bool:
    """
    """
    # check nan
    is_nan = np.any([
        np.any(np.isnan(param))
        for param in glove.embeddings_.values()
    ])

    # check inf
    is_inf = np.any([
        np.any(np.isnan(param))
        for param in glove.embeddings_.values()
    ])

    return any([is_nan, is_inf])


def prep_dataset(window_size_factor2: int,
                 data_fns: list[str],
                 eval_type: str,
                 eval_set_path: str) -> tuple[sp.coo_matrix, EvaluationSet, GloVeData]:
    """
    """
    # load data
    win_sz = window_size_factor2 * 2 - 1
    corpus = load_corpus(data_fns[win_sz])

    # prepare the evaluation
    if eval_type == 'split':
        train, valid, test = split_data(corpus.mat.tocoo())
        valid = valid + test

    else:
        train = corpus.mat
        valid = load_eval_dataset(eval_set_path)

    return train, valid, corpus


def fit(train_data: sp.coo_matrix,
        solver: str, n_components_log2: int,
        n_iters: int, alpha: float, x_max: float,
        lr_or_l2: float, init:float, share_params: bool) -> GloVe:
    """
    """
    # initiate and fit model
    d = int(2**n_components_log2)

    optional_params = dict()
    if solver == 'als':
        optional_params['l2'] =lr_or_l2
    else:  # sgd
        optional_params['learning_rate'] = lr_or_l2

    glove = GloVe(
        n_components=d,
        n_iters=n_iters,
        alpha=alpha,
        x_max=x_max,
        init=init,
        solver=solver,
        dtype=np.float32,
        share_params=share_params,
        num_threads=NUM_THREADS,
        random_state=RAND_STATE,
        **optional_params
    )
    glove.fit(train_data, verbose=True)

    return glove


def _objective(params: HyperParams,
               data_fns: list[str],
               eval_type: str,
               eval_set_path: str,
               solver: str = 'als',
               failure_score: float = 1e+3) -> float:
    """
    """
    # parse params
    (window_size_factor2, n_components_log2,
     n_iters, alpha, x_max, init, share_params, lr_or_l2) = params

    # prep data and fit the model
    train, valid, corpus = prep_dataset(window_size_factor2,
                                        data_fns,
                                        eval_type,
                                        eval_set_path)
    glove = fit(train, solver, n_components_log2, n_iters,
                alpha, x_max, lr_or_l2, init, share_params)

    # check the model fit failed (numerically)
    if is_model_bad(glove) or not hasattr(glove, 'embeddings_'):
        # to force the tuner think this is VERY BAD case
        # but this is very ad-hoc
        return failure_score

    # evaluate
    if eval_type == 'split':
        score = -glove.score(valid, weighted=False)  # MSE (the lower the better)
    else:
        predictions = compute_similarities(glove, corpus, valid,
                                           corpus._tokenizer.get_vocab())
        scores = compute_scores(valid, predictions)
        score = np.mean([v['tau'] for k, v in scores.items()])

    if np.isnan(score) or np.isinf(score):
        return failure_score

    return -score


def extract_argparse():
    """
    """
    # process the arguments
    parser = argparse.ArgumentParser(
        description="Fit and select GloVe model for given corpus"
    )

    parser.add_argument('data_path', type=str,
                        help='path for the dir contains pre-processed datasets')

    parser.add_argument('out_path', type=str,
                        help='path of the resulting search result and model')

    parser.add_argument('--solver', type=str, default='als',
                        choices={'als', 'sgd'},
                        help='solver for the GloVe model')

    parser.add_argument('--eval-set', type=str, default='split',
                        choices={'split', 'faruqui'})

    parser.add_argument('--eval-set-path', type=str, default=EVAL_DATA_PATH,
                        help='path where the evaluation data (`Faruqui`) located')

    parser.add_argument('--n-calls', type=int, default=100,
                        help='number of iteration (sampling) for the'
                             'Bayesian parameter search')

    parser.add_argument('--data-filename-template', type=str,
                        default='mxm_ws{window_size:d}_fold0.pkl',
                        help='dataset filename template')

    return parser.parse_args()


def main():
    """
    """
    args = extract_argparse()

    # load data filenames
    fn_tmp = args.data_filename_template
    fns = {
        i:join(args.data_path, fn_tmp.format(window_size=i))
        for i in range(3, 15, 2)
    }

    # hyper-parameter tuning: search the best model
    if args.solver == 'sgd':
        # if the solver is SGD, add `learn_rate`
        space = SPACE + [Real(1e-3, 1e-1, 'log-uniform', name='lr')]
    else:
        # if the solver is ALS, add `l2` regularization coefficient
        space = SPACE + [Real(1e-8, 2e+1, 'log-uniform', name='l2')]

    res_gp = gp_minimize(
        partial(_objective,
                data_fns=fns,
                eval_type=args.eval_set,
                eval_set_path=args.eval_set_path,
                solver=args.solver),
        space,
        n_calls=args.n_calls,
        random_state=RAND_STATE,
        verbose=True
    )

    # save search result
    skopt.dump(res_gp,
               join(args.out_path, 'search_result.skopt'),
               store_objective=False)

    # fit final model
    (window_size_factor2, n_components_log2,
     n_iters, alpha, x_max, init, share_params, lr_or_l2) = res_gp['x']  # optimal setup

    # load data and fit the model
    win_sz = window_size_factor2 * 2 - 1
    corpus = load_corpus(fns[win_sz])
    glove = fit(corpus.mat, args.solver, n_components_log2, n_iters,
                alpha, x_max, lr_or_l2, init, share_params)

    # save the results to disk
    glove.save(join(args.out_path, 'model.glv.mdl'))


if __name__ == "__main__":
    main()
