import os
from os.path import join
import json
import glob
import argparse
from typing import Optional, TypedDict, Union
from functools import partial
import logging

try:
    import skopt
except ModuleNotFoundError:
    print('[ERROR] `scikit-optimize` is not found in the python environment! '
          'install the package in your environment manually or '
          're-install `gloves` with `optimize` extra keyword. '
          '(i.e., pip install gloves[optimize])')
    print('Exiting...')
    raise SystemExit()

from skopt.space import Real, Integer, Categorical
from skopt.space.space import Dimension
from skopt.utils import use_named_args, point_asdict
from skopt import gp_minimize

import numpy as np
from scipy import sparse as sp

from gloves.solvers import als, sgd, ials
from gloves.evaluation import (split_data,
                               compute_similarities,
                               compute_scores,
                               EvaluationSet)
from gloves.corpus import Corpus, load_corpus
from gloves.utils import load_faruqui_wordsim_evalset as load_eval_dataset
from gloves.files import default_optimize_config

from .fit import fit_model


logger = logging.getLogger('Optimize')

RAND_STATE = os.environ.get('GLOVES_OPTIMIZE_RAND_STATE')
if RAND_STATE is not None:
    RAND_STATE = int(RAND_STATE)


FULL_DIMENSIONS = [
    'window_size_factor2',
    'n_components_log2',
    'n_iters',
    'alpha',
    'x_max',
    'l2',
    'lr',
    'share_params',
    'solver'
]


def read_optimize_config(config_fn: Optional[str]=None,
                         use_package_default: bool=False) -> list[Dimension]:
    """
    NOTE: currently, window size is not optional and will be added if not given
          also, the range and the interval also are fixed.

    arguments:
        config_fn: filename (path) to the custom search config file
                   if not given, using default config
        use_package_default: if True, uses factory defaults for any MISSING params
                             from `defaults` field of custom config file
    """
    # read input file & default file
    if config_fn is None:
        config_fn = default_optimize_config()

    with open(config_fn) as f:
        config = json.load(f)

    with open(default_optimize_config()) as f:
        default_config = json.load(f)

    # TODO: data validation step

    search_space = []
    defaults = {}
    search_targets = set(config['search_targets'])
    if 'solver' in search_targets:
        logger.warning("'solver' is included in search targets. "
                       "it will make the search process very inefficient. "
                       "consider increasing the number of calls.")

    if 'window_size_factor2' not in search_targets:
        logger.warning("'window_size_factor2' is not included, which currently "
                       "is not an optional one. We're adding default range...")
        search_targets.add('window_size_factor2')

    for name in FULL_DIMENSIONS:

        # fill defaults if it's not in search targets
        if name not in search_targets:
            default = config['defaults'].get(name)

            # if not specified, falls back to the package-wide default
            if default is None or use_package_default:

                if use_package_default:
                    default = default_config['defaults'][name]

            if name in 'window_size_factor2':
                defaults[name] = 2 * default - 1
            else:
                defaults[name] = default

        # fill the space container if we need to learn them
        else:
            spec = config['ranges'].get(name)
            # use default range
            if spec is None:
                spec = default_config['ranges'][name]

            if spec['type'] == 'integer':
                dimension = Integer(spec['low'], spec['high'], name=name)
            elif spec['type'] == 'real':
                dimension = Real(spec['low'], spec['high'], spec['prior'], name=name)
            elif spec['type'] == 'categorical':
                dimension = Categorical(spec['categories'], name=name)
            search_space.append(dimension)

    # if solver is fixed parameter, filter out search space suitable for the solver
    if 'solver' not in search_targets:
        if defaults['solver'] == 'sgd':
            search_space = [dim for dim in search_space
                            if dim.name not in sgd.UNSEARCHABLES]

        elif defaults['solver'] == 'als':
            search_space = [dim for dim in search_space
                            if dim.name not in als.UNSEARCHABLES]

        elif defaults['solver'] == 'ials':
            search_space = [dim for dim in search_space
                            if dim.name not in ials.UNSEARCHABLES]
        else:
            raise ValueError('[ERROR] only `als` and `sgd` solver are supported!')

    return search_space, defaults


def prep_dataset(window_size_factor2: int,
                 data_fns: list[str],
                 eval_type: str) -> tuple[sp.coo_matrix, EvaluationSet, Corpus]:
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
        valid = load_eval_dataset()

    return train, valid, corpus


def _objective(params: dict,
               fixed_params: dict,
               data_fns: list[str],
               search_space: list[Dimension],
               eval_type: str,
               num_threads: int=1,
               failure_score: float = 1e+3) -> float:
    """
    """
    params = {dim.name: param for dim, param in zip(search_space, params)}
    # fill the ones that are not searchable
    params.update(fixed_params)

    # prep data and fit the model
    train, valid, corpus = prep_dataset(params['window_size_factor2'],
                                        data_fns,
                                        eval_type)
    glove = fit_model(train, num_threads=num_threads, **params)

    # check the model fit failed (numerically)
    if glove.is_unhealthy or not hasattr(glove, 'embeddings_'):
        # to force the tuner think this is VERY BAD case
        # but this is very ad-hoc
        return failure_score

    # evaluate
    if eval_type == 'split':
        score = -glove.score(valid, weighted=False)  # MSE (the lower the better)
    else:
        predictions = compute_similarities(glove, corpus._tokenizer, valid,
                                           corpus._tokenizer.get_vocab())
        scores = compute_scores(valid, predictions)
        score = np.mean([v['corr'] for k, v in scores.items()])

    if np.isnan(score) or np.isinf(score):
        return failure_score

    return -score


def optimize(args):
    """
    arguments:
        data_path: directory where cooccurrence datasets are stored
        config: custom search configuration file (.json)
        out_path: path to save search result and final model
        eval_set: validation target {'split', 'faruqui'}
        n_calls: the number of iteration of Bayesian search
        data_filename_template: dataset filename template
                                (for various window sizes)
    """
    # load the config
    search_space, defaults = read_optimize_config(args.config)

    # load data filenames
    # TODO: this part is actually pain in the back (in usability perspective)
    #       although computing cooccurrence on the fly is too much inefficient
    fn_tmp = args.data_filename_template
    fns = {
        i:join(args.data_path, fn_tmp.format(window_size=i))
        for i in range(3, 15, 2)
    }

    res_gp = gp_minimize(
        partial(_objective,
                fixed_params=defaults,
                search_space=search_space,
                data_fns=fns,
                eval_type=args.eval_set,
                num_threads=args.num_threads),
        search_space,
        n_calls=args.n_calls,
        random_state=RAND_STATE,
        verbose=True
    )

    # save search result
    skopt.dump(res_gp,
               join(args.path, 'search_result.skopt'),
               store_objective=False)

    # fit final model with the optimal setup
    params = {dim.name: param for dim, param in zip(search_space, res_gp['x'])}
    params.update(defaults)

    # load data and fit the model
    win_sz = params['window_size_factor2'] * 2 - 1
    corpus = load_corpus(fns[win_sz])
    glove = fit_model(corpus.mat, num_threads=args.num_threads, **params)

    # save the results to disk
    glove.save(join(args.path, args.out))
