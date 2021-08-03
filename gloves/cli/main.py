import os
from os.path import join, basename, splitext
from functools import partial
from typing import Optional
from collections import defaultdict
import argparse
import logging
import multiprocessing as mp
import pickle as pkl
import time
from tqdm import tqdm

from .learntoken import main as fit_tokenizer
from .fit import opthyper, fit
from ..corpus import load_corpus
from ..model import GloVe
from ..evaluation import compute_scores, compute_similarities
from ..utils import load_faruqui_wordsim_evalset, init_tokenizer


logging.basicConfig()
logger = logging.getLogger("GloVesCLI")


def parse_arguments():
    """
    """
    parser = argparse.ArgumentParser(
        prog="GloVes Training Utilities",
        description="Prepare / train GloVe models"
    )
    subparsers = parser.add_subparsers(title="command",
                                       dest="command",
                                       help="sub-command help")
    subparsers.required = True

    # define comman shared arguments
    base_subparser = argparse.ArgumentParser(add_help=False)

    base_subparser.add_argument("-o", "--out", type=str,
                                default='output.pkl',
                                help="output filename.")

    base_subparser.add_argument("-p", "--path", type=str, default="./",
                                help="path where the output stored")

    base_subparser.add_argument("--quiet", default=True,
                                action=argparse.BooleanOptionalAction,
                                help="set verbosity")

    base_subparser.add_argument('-j', '--num-threads', type=int, default=1,
                                help=('number of cores to be used for the parallelism. '
                                      'only used for `optimize` and `train`'))


    # `tokenizer` sub command =================================================
    tokenizer = subparsers.add_parser('tokenizer',
                                      parents=[base_subparser],
                                      help='fit tokenizer')

    tokenizer.add_argument("textfile", type=str,
                           help="text file to be fit tokenizer")

    tokenizer.add_argument("-n", "--num-tokens", type=int, default=20_000,
                           help="number of tokens to be fitted")

    # `optimize` sub command =================================================
    optimize = subparsers.add_parser('optimize',
                                      parents=[base_subparser],
                                      help='hyper-parameter search for GloVe models')

    optimize.add_argument('data_path', type=str,
                          help='path for the dir contains pre-processed datasets')

    optimize.add_argument('out_path', type=str,
                          help='path of the resulting search result and model')

    optimize.add_argument('--config', type=str, default=None,
                          help='path for the configuration file')

    optimize.add_argument('--n-calls', type=int, default=100,
                          help='number of iteration (sampling) for the'
                               'Bayesian parameter search')

    optimize.add_argument('--eval-set', type=str, default='faruqui',
                          choices={'split', 'faruqui'})

    optimize.add_argument('--data-filename-template', type=str,
                          default='corpus_ws{window_size:d}.cooccur.pkl',
                          help='dataset filename template')

    # `train` sub command =================================================
    train = subparsers.add_parser('train',
                                  parents=[base_subparser],
                                  help='train a GloVe model')

    train.add_argument('data', type=str,
                       help='path for the pre-processed cooccurrence dataset')

    train.add_argument('out_path', type=str,
                       help='path of the resulting search result and model')

    train.add_argument('--n-components', type=int, default=32,
                       help='size of latent dimenions')

    train.add_argument('--solver', type=str, default='sgd',
                       choices={'als', 'sgd'},
                       help='optimization method {"als", "sgd"}')

    train.add_argument('--n-iters', type=int, default=15,
                       help='number of epoches')

    train.add_argument('--alpha', type=float, default=0.75,
                       help='controls the smoothness of count measurement')

    train.add_argument('--x-max', type=float, default=100,
                       help=('set the normalization factor (threshold) of '
                             'raw count measurement'))

    train.add_argument('--l2', type=float, default=1e-4,
                       help='L2 regularzation coefficient (only for ALS solver)')

    train.add_argument('--init', type=float, default=1e-2,
                       help='initialization weight (only for ALS solver)')

    train.add_argument('--learning-rate', type=float, default=1e-2,
                       help='learning rate (only for SGD solver)')

    train.add_argument("--share-params", default=True,
                       action=argparse.BooleanOptionalAction,
                       help="determine use 2 sets of embeddings or one for words")

    # `evaluate` sub command ===============================================
    evaluate = subparsers.add_parser('evaluate',
                                     parents=[base_subparser],
                                     help='evaluate a trained GloVe model')

    evaluate.add_argument('tokenizer', type=str,
                          help='path for the tokenizer dump (.json)')

    evaluate.add_argument('model', type=str,
                          help='path for the trained GloVe model')

    return parser.parse_args()


def main():
    """
    """
    args = parse_arguments()

    if args.command == 'tokenizer':
        fit_tokenizer(args)
    elif args.command == 'optimize':
        opthyper(args)
    elif args.command == 'train':
        fit(args)
    elif args.command == 'evaluate':

        # load model
        glove = GloVe.from_file(args.model)
        tokenizer = init_tokenizer(path=args.tokenizer)
        glove.set_tokenizer(tokenizer)

        valid = load_faruqui_wordsim_evalset()
        preds = compute_similarities(glove, tokenizer, valid,
                                     tokenizer.get_vocab())
        scores = compute_scores(valid, preds)
        score = sum([v['corr'] for k, v in scores.items()]) / len(scores)

        # print results
        print()
        print(f'[Word-sim Averaged Correlation]: {score:.4f}')
        print('=' * (15 + 6 + 6 + 10 + 8))
        print(f"{'dataset':<15}  {'rho':<6}  {'p':<6}  {'missing(%)':<6}")
        print('=' * (15 + 6 + 6 + 10 + 8))
        for dataset, result in scores.items():
            print(f"{dataset:<15}  {result['corr']:.4f}  " +
                  f"{result['p']:.4f}  {result['nan_rate']:.2%}")
        print()

    else:
        ValueError('[ERROR] only `tokenizer`, `optimize` '
                   'and `train` are supported!')
