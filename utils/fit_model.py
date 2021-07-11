import os
from os.path import dirname, basename, join, splitext
from typing import Optional, TypedDict

os.environ['MKL_NUM_THREADS'] = '1'
if 'NUMBA_NUM_THREADS' not in os.environ:
    os.environ['NUMBA_NUM_THREADS'] = '4'

import sys
sys.path.append(join(dirname(__file__), '..'))

import json
import glob
import argparse
from functools import partial
import pickle as pkl

import numpy as np
from scipy import sparse as sp
from scipy.stats import kendalltau

import skopt
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

from tokenizers import Tokenizer

from gloveals.models.als import GloVeALS
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
    tokens_inv: dict[str, int]
    tokenizer: Optional[str]
    spec: GloVeDataSpec

# alias for the score object
class EvaluationScore(TypedDict):
    tau: float
    p: float
    nan_rate:float

# alias for the other data objects
EvaluationSet = dict[str, dict[frozenset, float]]
EvaluationResult = dict[str, EvaluationScore]
Predictions = dict[str, tuple[str, str]]

# ============== Data Object Definitions =================


RAND_STATE = 1234
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
    Integer(2, 8, name='n_components_log2'),
    Real(1e-4, 2e+1, 'log_uniform', name='l2'),
    Real(1e-4, 1e-1, 'log_uniform', name='init'),
    Integer(10, 80, name='n_iters'),
    Real(0.5, 1, name='alpha'),
    Real(1e+1, 1e+2, 'log_uniform', name='x_max')
]


def load_data(data_fn: str) -> GloVeData:
    """
    """
    with open(data_fn, 'rb') as f:
        data = pkl.load(f)
        tokenizer_param = data.get('tokenizer_param')  # json (serialized)
        if tokenizer_param is not None:
            data['tokenizer'] = Tokenizer.from_str(
                json.dumps(data['tokenizer_dict'])
            )
        else:
            data['tokenizer'] = None
    return data


def load_eval_dataset(path: str) -> EvaluationSet:
    """
    """
    valid_fns = glob.glob(join(path, '*.txt'))

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
        tok = tokenizer.encode([token])
        if len(tok.tokens) == 1:
            return tok.ids[0]
        else:
            return None

    # if tokenizer is not given, use the token inverse map
    if token in token_inv_map:
        return token_inv_map[toke]

    # otherwise, return None, as indication of `not-found`
    return None


def compute_similarities(glove: GloVeALS,
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
            w1_exists = check_exists(w1, data['tokens_inv'], data['tokenizer'])
            w2_exists = check_exists(w2, data['tokens_inv'], data['tokenizer'])
            if w1_exists is None or w2_exists is None:
                predictions[dataset][pair] = None
                continue

            i1 = data['tokens_inv'][w1]
            i2 = data['tokens_inv'][w2]
            predictions[dataset][pair] = W[i1] @ W[i2]

    return predictions


def compute_scores(eval_set: EvaluationSet,
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


def is_model_bad(glove: GloVeALS) -> bool:
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


def _objective(params: tuple[float],
               data_fns: list[str],
               eval_type: str,
               failure_score: float = 1e+3) -> float:
    """
    """
    # parse params
    (window_size_factor2,
     n_components_log2,
     l2, init, n_iters, alpha, x_max) = params

    # load data
    win_sz = window_size_factor2 * 2 - 1
    data = load_data(data_fns[win_sz])

    # prepare the evaluation
    if eval_type == 'split':
        Xtr, Xvl, Xts = split_data(data['X'].tocoo())
        Xvl = Xvl + Xts

    else:
        Xtr = data['X']
        eval_set = load_eval_dataset(EVAL_DATA_PATH)

    # initiate and fit model
    d = int(2**n_components_log2)
    glove = GloVeALS(
        n_components=d,
        l2=l2,
        init=init,
        n_iters=n_iters,
        alpha=alpha,
        x_max=x_max,
        dtype=np.float32
    )
    glove.fit(Xtr)

    # check the model fit failed (numerically)
    if is_model_bad(glove) or not hasattr(glove, 'embeddings_'):
        # to force the tuner think this is VERY BAD case
        # but this is very ad-hoc
        return failure_score

    # evaluate
    if eval_type == 'split':
        score = -glove.score(Xvl, weighted=False)
    else:
        predictions = compute_similarities(glove, data, eval_set)
        scores = compute_scores(eval_set, predictions)
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
                        help='path where the model (.pkl) is stored')

    parser.add_argument('--eval-set', type=str, default='split',
                        choices={'split', 'faruqui'})

    parser.add_argument('--n-calls', type=int, default=100,
                        help='number of iteration (sampling) for the'
                             'Bayesian parameter search')

    return parser.parse_args()


def main():
    """
    """
    args = extract_argparse()

    # load data filenames
    fn_tmp = 'msd_mxm_w2v_cooccur_uncased_winlen{:d}.pkl'
    fns = {
        i:join(args.data_path, fn_tmp.format(i))
        for i in range(1, 13, 2)
    }

    # hyper-parameter tuning: search the best model
    res_gp = gp_minimize(
        partial(_objective, data_fns=fns, eval_type=args.eval_set),
        SPACE,
        n_calls=args.n_calls,
        random_state=RAND_STATE,
        verbose=True
    )

    # save
    bn = fn_tmp.split('_winlen')[0]
    out_fn = join(args.out_path, bn + '.pkl')
    skopt.dump(res_gp, out_fn)


if __name__ == "__main__":
    main()
