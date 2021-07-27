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

from gloveals.models.base import GloVeBase
from gloveals.models.als import GloVeALS
from gloveals.models.sgd import GloVeSGD
from gloveals.evaluation import split_data


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
HyperParams = tuple[int, int, int, float, float, bool, float]

# ============== Data Object Definitions =================

NUM_THREADS = int(os.environ['NUMBA_NUM_THREADS'])
RAND_STATE = 1234
# np.random.seed(RAND_STATE)

EVAL_DATA_PATH = join(
    dirname(__file__),
    '../eval-word-vectors/data/word-sim/'
)
# EVAL_FNS = glob.glob(join(EVAL_DATA_PATH, '*.txt'))
# if len(EVAL_FNS) == 0:
#     raise ValueError('[ERROR] cannot find evaluation files!')

# setup the search space
SPACE = [
    Integer(1, 6, name='window_size_factor2'),
    Integer(4, 9, name='n_components_log2'),
    Integer(10, 80, name='n_iters'),
    Real(0.5, 1, name='alpha'),
    Real(1e+1, 1e+2, 'log_uniform', name='x_max'),
    Categorical([True, False], name='share_params')
]


def load_data(data_fn: str) -> GloVeData:
    """
    """
    with open(data_fn, 'rb') as f:
        data = pkl.load(f)

        # convert triplet list to the matrix
        row = data['triplets']['row']
        col = data['triplets']['col']
        val = data['triplets']['data']
        n_entities = len(data['token_inv_map'])
        shape = (n_entities, n_entities)
        data['X'] = sp.coo_matrix((val, (row, col)), shape=shape)

        # equip tokenizer
        tokenizer_param = data.get('tokenizer_params')  # json (serialized)
        if tokenizer_param is not None:
            data['tokenizer'] = Tokenizer.from_str(tokenizer_param)
        else:
            data['tokenizer'] = None

    return data


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


def compute_similarities(glove: GloVeBase,
                         data: GloVeData,
                         eval_set: EvaluationSet,
                         tokenizer: Optional[Tokenizer] = None) -> Predictions:
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
            i1 = check_exists(w1, data['token_inv_map'], data['tokenizer'])
            i2 = check_exists(w2, data['token_inv_map'], data['tokenizer'])
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


def is_model_bad(glove: GloVeBase) -> bool:
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
    data = load_data(data_fns[win_sz])

    # prepare the evaluation
    if eval_type == 'split':
        train, valid, test = split_data(data['X'].tocoo())
        valid = valid + test

    else:
        train = data['X']
        valid = load_eval_dataset(eval_set_path)

    return train, valid, data


def fit(train_data: sp.coo_matrix,
        solver: str, n_components_log2: int,
        n_iters: int, alpha: float, x_max: float,
        lr_or_l2: float, share_params: bool) -> GloVeBase:
    """
    """
    # initiate and fit model
    d = int(2**n_components_log2)

    if solver == 'als':
        glove = GloVeALS(
            n_components=d,
            l2=lr_or_l2,
            n_iters=n_iters,
            alpha=alpha,
            x_max=x_max,
            dtype=np.float32,
            share_params=share_params,
            num_threads=NUM_THREADS,
            random_state=RAND_STATE
        )
    else:  # sgd
        glove = GloVeSGD(
            n_components=d,
            learning_rate=lr_or_l2,
            n_iters=n_iters,
            alpha=alpha,
            x_max=x_max,
            dtype=np.float32,
            share_params=share_params,
            num_threads=NUM_THREADS,
            random_state=RAND_STATE
        )
    glove.fit(train_data)

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
     n_iters, alpha, x_max, share_params, lr_or_l2) = params

    # prep data and fit the model
    train, valid, data = prep_dataset(window_size_factor2,
                                      data_fns,
                                      eval_type,
                                      eval_set_path)
    glove = fit(train, solver, n_components_log2, n_iters,
                alpha, x_max, lr_or_l2, share_params)

    # check the model fit failed (numerically)
    if is_model_bad(glove) or not hasattr(glove, 'embeddings_'):
        # to force the tuner think this is VERY BAD case
        # but this is very ad-hoc
        return failure_score

    # evaluate
    if eval_type == 'split':
        score = -glove.score(valid, weighted=False)
    else:
        predictions = compute_similarities(glove, data, valid)
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
                        default='lyrics_fold0_ws{window_size:d}.glove_data.pkl',
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
        for i in range(1, 13, 2)
    }

    # hyper-parameter tuning: search the best model
    if args.solver == 'sgd':
        # if the solver is SGD, add `learn_rate`
        space = SPACE + [Real(1e-3, 1e+0, 'log_uniform', name='lr')]
    else:
        # if the solver is ALS, add `l2` regularization coefficient
        space = SPACE + [Real(1e-8, 2e+1, 'log_uniform', name='l2')]

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
     n_iters, alpha, x_max, share_params, lr_or_l2) = res_gp['x']  # optimal setup

    # load data and fit the model
    win_sz = window_size_factor2 * 2 - 1
    data = load_data(fns[win_sz])
    glove = fit(data['X'], args.solver, n_components_log2, n_iters,
                alpha, x_max, lr_or_l2, share_params)

    # save the results to disk
    params_to_save = {
        'W': glove.embeddings_['W'],
        'bi': glove.embeddings_['bi']
    }
    if not share_params:
        params_to_save.update({
            'H': glove.embeddings_['H'],
            'bj': glove.embeddings_['bj']
        })
    np.savez(join(args.out_path, 'model.npz'), **params_to_save)


if __name__ == "__main__":
    main()
