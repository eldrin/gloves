import argparse
import logging
from os.path import join

from ..corpus import Corpus


logging.basicConfig()
logger = logging.getLogger("Cooccurrence")


def parse_arguments():
    """
    """
    parser = argparse.ArgumentParser(
        description="Preprocess the corpus data for GloVe trainig"
    )

    parser.add_argument("textfile", type=str,
                        help="text file to be processed")

    parser.add_argument("-o", "--out", type=str, default=None,
                        help="output filename")

    parser.add_argument("-p", "--path", type=str, default="./",
                        help="path where the output stored")

    parser.add_argument("-w", "--window-size", type=int, default=10,
                        help="window size for the cooccurrence check")

    parser.add_argument("-t", "--tokenizer", type=str, default=None,
                        help="filename of the custom `huggingface` tokenizer model")

    parser.add_argument("--weighted-count", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="use distance based weighting on cooccurrence score")

    parser.add_argument("--symmetrization", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="decide whether cooccurrence matrix is symmetrized")

    parser.add_argument("--quiet", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")

    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.quiet:
        logger.setLevel(logging.INFO)

    logger.info("[Processing cooccurrence matrix]")

    # instantiate corpus processor
    corpus = Corpus(
        args.window_size,
        not args.weighted_count,
        args.tokenizer
    )

    # loading the text filepoint
    logger.info("-- Building matrix...")
    with open(args.textfile) as fp:
        corpus.build_matrix(fp, symmetrization=args.symmetrization)
    logger.info("finished!")

    # save the output
    logger.info("-- Saving data...")
    out_fn = 'cooccurrence.pkl' if args.out is None else args.out
    corpus.save(join(args.path, out_fn))
    logger.info("finished!")
