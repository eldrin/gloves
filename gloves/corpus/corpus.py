import os
from os.path import exists
from typing import Optional, Union
import pickle as pkl
from collections import defaultdict
import logging
import mmap

from scipy import sparse as sp
from tokenizers import Tokenizer
from tqdm import tqdm

from ..utils import init_tokenizer, count_lines


logger = logging.getLogger(__name__ + '.corpus')


class Corpus:
    def __init__(self,
                 window_size: int=10,
                 uniform_count: bool=False,
                 tokenizer_path: Optional[str]=None,
                 dtype: str='float32') -> None:
        """
        """
        self.window_size = window_size
        self.uniform_count = uniform_count
        self.dtype = dtype

        # if tokenizer is not provided, falls back to the default
        # tokenizer, which trained on the English Wikipedia dump (dumped at 2021/03)
        self._tokenizer = init_tokenizer(tokenizer_path)

    @property
    def n_tokens(self) -> int:
        return self._tokenizer.get_vocab_size()

    def build_matrix(self,
                     text: Union[list[str], str],
                     symmetrization: bool=False,
                     verbose: bool=False) -> sp.coo_matrix:
        """
        both file pointer for the text file and pre-load list of strings
        can be fed to this function

        TODO: push the `astype` part into the "compute cooccurrence" function
        """
        self.mat = compute_cooccurrence(text,
                                        self._tokenizer,
                                        self.window_size,
                                        self.uniform_count,
                                        symmetrization,
                                        verbose).astype(self.dtype)
        return self.mat

    def from_file(self, corpus_fn: str) -> None:
        """ load the dumped (processed) corpus file to the memory
        """
        with open(corpus_fn, 'rb') as fp:
            data = pkl.load(fp)

        self._tokenizer = init_tokenizer(dump_str=data['tokenizer'])
        self.mat = sp.coo_matrix(
            (data['mat']['counts'], (data['mat']['row'], data['mat']['col'])),
            shape = (self.n_tokens, self.n_tokens),
            dtype = self.dtype
        )
        self.window_size = data['window_size']
        self.uniform_count = data['uniform_count']

    def save(self, out_fn: str) -> None:
        """ save the processed corpus file to the disk
        """
        # we separate the tokenizer and corpus, if custom tokenizer is used
        logger.info('Saving corpus...')
        with open(out_fn, 'wb') as fp:
            pkl.dump(
                {
                    'mat': {
                        'row': self.mat.row,
                        'col': self.mat.col,
                        'counts': self.mat.data
                    },
                    'tokenizer': self._tokenizer.to_str(),
                    'uniform_count': self.uniform_count,
                    'window_size': self.window_size
                },
                fp
            )


def compute_cooccurrence(path_or_lines: Union[str, list[str]],
                         tokenizer: Tokenizer,
                         window_size: int = 10,
                         uniform_count: bool = False,
                         symmetrization: bool = True,
                         verbose: bool = False,
                         tqdm_position: int = 0) -> sp.coo_matrix:
    """
    NOTE:
    tqdm_position is used only for the visualization purpose for
    the CLI application. it's never relevant for usual API usage
    """
    cooccur = dict()

    is_text_file = False
    if isinstance(path_or_lines, str) and exists(path_or_lines):
        is_text_file = True
        fp = open(path_or_lines, "r+b")
        mm = mmap.mmap(fp.fileno(), 0, prot=mmap.PROT_READ)
        lines = iter(mm.readline, b"")
    else:
        lines = path_or_lines

    if verbose:
        # get the line number of the text to check progress
        if is_text_file:
            num_lines = count_lines(path_or_lines)
        else:
            num_lines = len(lines)
    else:
        num_lines = 1

    # 1. Read and tokenize
    text = f"job #{tqdm_position+1:d}"
    with tqdm(total=num_lines, ncols=80, desc=text,
              disable=not verbose, position=tqdm_position+1) as prog:
        for line in lines:
            # strip line break
            line = line.decode('utf8').replace('\n', '')

            # tokenize
            token_ids = tokenize(line, tokenizer)
            update_cooccurrence(token_ids, cooccur, window_size, uniform_count)

            prog.update()

    if is_text_file:
        mm.close()
        fp.close()

    # convet cooccur matrix to coo matrix
    n_tokens = tokenizer.get_vocab_size()
    mat = cooccur2spmat(cooccur, n_tokens, symmetrization)

    return mat


def tokenize(line: str,
             tokenizer: Tokenizer,
             max_len: int=1000,
             batch_size: int=50) -> list[int]:
    """
    """
    if len(line) > max_len:
        output = []
        naive_tokens = line.split(' ')
        for i in range(0, len(naive_tokens), batch_size):
            batch = ' '.join(naive_tokens[i:i+batch_size])
            tok = tokenizer.encode(batch)
            output.extend(tok.ids)
        return output

    else:
        tok = tokenizer.encode(line)
        return tok.ids


def update_cooccurrence(token_ids: list[int],
                        cooccur: dict[int, dict[int, float]],
                        window_size: int = 10,
                        uniform_count: bool = False) -> None:
    """
    this maybe the bottleneck.
    TODO: would numba or cython will do better job?
    """
    glb_end = len(token_ids)
    for i in range(glb_end):
        cur = token_ids[i]

        lcl_end = min(i + window_size + 1, glb_end)
        for j in range(i+1, lcl_end):
            other = token_ids[j]

            if cur <= other:
                row, col = cur, other
            else:
                row, col = other, cur

            if row not in cooccur:
                cooccur[row] = defaultdict(float)

            if col not in cooccur[row]:
                cooccur[row][col] = 0.

            # prep contiainor if not yet made
            count = 1. if uniform_count else (1. / (j - i))
            cooccur[row][col] += count


def cooccur2spmat(cooccur: dict[int, dict[int, float]],
                  n_tokens: int,
                  symmetrization: bool=True) -> sp.coo_matrix:
    """
    """
    shape = (n_tokens, n_tokens)
    I, J, V = [], [], []
    for cur, neighbors in cooccur.items():
        for other, count in neighbors.items():
            I.append(cur)
            J.append(other)
            V.append(count)
    mat = sp.coo_matrix((V, (I, J)), shape=shape)

    # to make sure the symmetry
    if symmetrization:
        mat_t = mat.T
        mat = (mat + mat_t).tocoo()

    # make sure there's no zeros
    mat.eliminate_zeros()

    return mat


def load_corpus(path: str) -> Corpus:
    """ simple wrapper for loading saved corpus
    """
    corpus = Corpus()
    corpus.from_file(path)
    return corpus
