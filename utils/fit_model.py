import os
from os.path import dirname, basename, join, splitext

os.environ['MKL_NUM_THREADS'] = '1'
if 'NUMBA_NUM_THREADS' not in os.environ:
    os.environ['NUMBA_NUM_THREADS'] = '2'

import sys
sys.path.append(join(dirname(__file__), '..'))
import glob
import argparse
import pickle as pkl

import numpy as np
from scipy.stats import kendalltau
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

from gloveals.models.als import GloVeALS
from gloveals.evaluation import split_data


RAND_STATE = 1234
EVAL_DATA_PATH = join(
    dirname(__file__),
    '../eval-word-vectors/data/word-sim/'
)
# EVAL_FNS = glob.glob(join(EVAL_DATA_PATH, '*.txt'))
# if len(EVAL_FNS) == 0:
#     raise ValueError('[ERROR] cannot find evaluation files!')


def load_data(data_fn):
    """
    """
    with open(data_fn, 'rb') as f:
        data = pkl.load(f)
    return data


def load_eval_dataset(path):
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


def compute_similarities(glove, data, eval_set):
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
            if w1 not in data['tokens_inv'] or w2 not in data['tokens_inv']:
                predictions[dataset][pair] = None
                continue

            i1 = data['tokens_inv'][w1]
            i2 = data['tokens_inv'][w2]
            predictions[dataset][pair] = W[i1] @ W[i2]
            
    return predictions


def compute_scores(eval_set, predictions):
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


def select_model(data, args, n_calls=100):
    """
    """
    # prepare the evaluation
    if args.eval_set == 'split':
        Xtr, Xvl, Xts = split_data(data['X'].tocoo())
        Xvl = Xvl + Xts
    
    else:
        Xtr = data['X']
        eval_set = load_eval_dataset(EVAL_DATA_PATH)
        
    # setup the search space
    space = [
        Integer(2, 10, name='n_components_log2'),
        Real(1e-10, 1e+4, 'log_uniform', name='l2'),
        Real(1e-3, 1, 'log_uniform', name='init'),
        Integer(10, 50, name='n_iters'),
        Real(0.5, 1, name='alpha'),
        Real(1e+1, 1e+3, 'log_uniform', name='x_max')
    ]
    
    @use_named_args(space)
    def objective(**params):
        """
        """
        # initiate and fit model
        d = int(2**params['n_components_log2'])
        glove = GloVeALS(
            n_components=d,
            l2=params['l2'],
            init=params['init'],
            n_iters=params['n_iters'],
            alpha=params['alpha'],
            x_max=params['x_max'],
            dtype=np.float32
        )
        glove.fit(Xtr)
        
        # evaluate
        if args.eval_set == 'split':
            score = -glove.score(Xvl, weighted=False)
        else:
            predictions = compute_similarities(glove, data, eval_set)
            scores = compute_scores(eval_set, predictions)
            score = np.mean([v['tau'] for k, v in scores.items()])

        return -score
    
    # search the best model
    res_gp = gp_minimize(objective,
                         space,
                         n_calls=n_calls,
                         random_state=RAND_STATE,
                         verbose=True)

    return res_gp


def extract_argparse():
    """
    """
    # process the arguments
    parser = argparse.ArgumentParser(
        description="Fit and select GloVe model for given corpus"
    )

    parser.add_argument('data_fn', type=str,
                        help='filename for the file contains pre-processed dataset')

    parser.add_argument('out_path', type=str,
                        help='path where the model (.pkl) is stored')
    
    parser.add_argument('--eval-set', type=str, default='split',
                        choices={'split', 'faruqui'})

    return parser.parse_args()


def main():
    """
    """
    args = extract_argparse()
    
    # load data
    data = load_data(args.data_fn)
    
    # hyper-parameter tuning
    res_gp = select_model(data, args)
    
    # save
    bn = splitext(basename(args.data_fn))[0]
    out_fn = join(args.out_path, bn + '.pkl')
    with open(out_fn, 'wb') as f:
        pkl.dump(res_gp, f)


if __name__ == "__main__":
    main()