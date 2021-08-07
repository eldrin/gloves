from typing import Optional
import argparse
import logging
import time
from os.path import join

from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import Tokenizer

from ..utils import init_tokenizer


logging.basicConfig()
logger = logging.getLogger("LearnTokenizer")


def parse_arguments():
    """
    """
    parser = argparse.ArgumentParser(
        prog="FitTokenizer",
        description="A simple wrapper fits huggingface BPE tokenzer"
    )

    parser.add_argument("textfile", type=str,
                        help="text file to be fit tokenizer")

    parser.add_argument("-n", "--num-tokens", type=int, default=20_000,
                        help="number of tokens to be fitted")

    parser.add_argument("-o", "--out", type=str,
                        default='output',
                        help="output filename.")

    parser.add_argument("-p", "--path", type=str, default="./",
                        help="path where the output stored")

    parser.add_argument("--quiet", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")

    return parser.parse_args()


def train_tokenizer(text_fn: str,
                    out_fn: Optional[str]=None,
                    vocab_size: int=300_000,
                    verbose: bool=True) -> Tokenizer:
    """
    """
    # initialize the tokenizer
    tokenizer = init_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=verbose,
        initial_alphabet=ByteLevel.alphabet()
    )

    # ========= training ==========
    t0 = time.time()
    tokenizer.train([text_fn], trainer)

    # ========= saving ===========
    if out_fn is not None:
        tokenizer.save(path=out_fn)

    if verbose:
        dur = time.time() - t0
        logger.info(f'\tTraining finished ({dur:.4f}s)')
        logger.info(f"\tTrained vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


def main():
    """
    """
    args = parse_arguments()

    verbose = not args.quiet
    if verbose:
        logger.setLevel(logging.INFO)

    logger.info(f"[Learning BPE tokenizer of {args.num_tokens:d} vocabs...]")

    # do the job
    train_tokenizer(text_fn    = args.textfile,
                    out_fn     = join(args.path, args.out),
                    vocab_size = args.num_tokens,
                    verbose    = verbose)
