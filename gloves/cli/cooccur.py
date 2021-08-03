import os
from os.path import join, basename, splitext
from functools import partial
from typing import Optional
from collections import defaultdict
import argparse
import logging
import multiprocessing as mp
import pickle as pkl
from tqdm import tqdm

from ..corpus import (Corpus,
                      compute_cooccurrence,
                      load_corpus)


logging.basicConfig()
logger = logging.getLogger("CooccurCLI")


COOCCUR_MAX_JOB = os.environ.get('COOCCUR_MAX_JOB')
if COOCCUR_MAX_JOB is None:
    COOCCUR_MAX_JOB = 8


def parse_arguments():
    """
    """
    parser = argparse.ArgumentParser(
        prog="Cooccurrence Matrix Builder",
        description="Preprocess the corpus data for GloVe trainig"
    )
    subparsers = parser.add_subparsers(title="command",
                                       dest="command",
                                       help="sub-command help")
    subparsers.required = True

    base_subparser = argparse.ArgumentParser(add_help=False)
    # define comman shared arguments

    base_subparser.add_argument("-p", "--path", type=str, default="./",
                                help="path where the output stored")

    base_subparser.add_argument("--symmetrization", default=True,
                                action=argparse.BooleanOptionalAction,
                                help="decide whether cooccurrence matrix is symmetrized. ")

    base_subparser.add_argument("--quiet", default=True,
                                action=argparse.BooleanOptionalAction,
                                help="set verbosity")

    # `build` sub command ========================================================
    build = subparsers.add_parser('build',
                                  parents=[base_subparser],
                                  help='build cooccurrence from text')

    build.add_argument("textfiles", type=str, nargs='+',
                       help="text file(s) to be processed")

    build.add_argument("-w", "--window-size", type=int, default=10,
                       help="window size for the cooccurrence check")

    build.add_argument("-t", "--tokenizer", type=str, default=None,
                       help="filename of the custom `huggingface` tokenizer model")

    build.add_argument("-j", "--num-processes", type=int, default=1,
                       help=("the number of processes to be launched for "
                             "processing textfiles. can't get larger than "
                             "the number of files given"))

    build.add_argument("--weighted-count", default=True,
                       action=argparse.BooleanOptionalAction,
                       help="use distance based weighting on cooccurrence score")


    # `merge` sub command ========================================================
    merge = subparsers.add_parser('merge',
                                  parents=[base_subparser],
                                  help='merge cooccurrence files')

    merge.add_argument("cooccurfiles", type=str, nargs='+',
                       help="cooccur file(s) to be processed")

    merge.add_argument("-o", "--out", type=str, default='merged_cooccur.pkl',
                       help="output filename")

    return parser.parse_args()


def process_file(pid: int,
                 filename: str,
                 out_path: str,
                 window_size: int = 10,
                 uniform_count: bool = False,
                 tokenizer: Optional[str] = None,
                 symmetrization: bool = True,
                 verbose: bool = False) -> None:
    """
    """
    # This line is the strange hack
    # (https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308)
    print(' ', end='', flush=True)

    # initiate corpus
    corpus = Corpus(window_size, uniform_count, tokenizer)

    # get cooccurrence matrix
    corpus.mat = compute_cooccurrence(filename,
                                      corpus._tokenizer,
                                      window_size,
                                      uniform_count,
                                      symmetrization,
                                      verbose,
                                      pid)

    # save the result
    fn = join(out_path, splitext(basename(filename))[0] + '.cooccur.pkl')
    corpus.save(fn)


def build_cooccur(args):
    """
    """
    # for windows support
    mp.freeze_support()

    if not args.quiet:
        logger.setLevel(logging.INFO)

    logger.info("[Processing cooccurrence matrix]")

    # loading the text filepoint
    # it's capped to `COOCCUR_MAX_JOB`, which can be overridden by
    # export COOCCUR_MAX_JOB=N
    n_jobs = min(min(len(args.textfiles), args.num_processes), COOCCUR_MAX_JOB)
    pool = mp.Pool(n_jobs, initargs=(tqdm.get_lock(),), initializer=tqdm.set_lock)
    for i, file in enumerate(args.textfiles):
        pool.apply_async(process_file,
                         args=(i,
                               file,
                               args.path,
                               args.window_size,
                               not args.weighted_count,
                               args.tokenizer,
                               args.symmetrization,
                               not args.quiet))
    pool.close()
    pool.join()
    if not args.quiet:
        print('\n' * (len(args.textfiles) + 1))

    logger.info("finished!")


def merge_cooccur(args):
    """
    we assume all partial data used the same tokenizer
    """
    if not args.quiet:
        logger.setLevel(logging.INFO)

    merged = dict.fromkeys(['mat', 'window_size', 'uniform_count'])
    with tqdm(total=len(args.cooccurfiles), ncols=80, disable=args.quiet) as prog:
        for file in args.cooccurfiles:
            # load the data
            corpus = load_corpus(file)

            if merged['window_size'] is None:
                merged['window_size'] = corpus.window_size

            if merged['uniform_count'] is None:
                merged['uniform_count'] = corpus.uniform_count

            mat = corpus.mat.astype('float32')
            if args.symmetrization:
                mat = (mat + mat.T.tocoo()).tocoo()

            if merged['mat'] is None:
                merged['mat'] = mat
            else:
                merged['mat'] += mat

            prog.update()
        merged['mat'] = merged['mat'].tocoo()

    # save output
    logger.info('Saving to disk...')
    out_fn = join(args.path, args.out)
    with open(out_fn, 'wb') as fp:
        pkl.dump(
            {
                'mat': {
                    'row': merged['mat'].row,
                    'col': merged['mat'].col,
                    'counts': merged['mat'].data
                },
                'uniform_count': merged['uniform_count'],
                'window_size': merged['window_size']
            },
            fp
        )


def main():
    """
    """
    args = parse_arguments()

    if args.command == 'build':
        build_cooccur(args)
    elif args.command == 'merge':
        merge_cooccur(args)
    else:
        ValueError('[ERROR] only `build` and `merge` are supported!')
