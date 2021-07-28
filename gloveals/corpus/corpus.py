from typing import Optional, Union, TextIO
import pickle as pkl

from scipy import sparse as sp
from tokenizers import Tokenizer

from ..utils import init_tokenizer


class Corpus:
    def __init__(self,
                 window_size: int=10,
                 uniform_count: bool=False,
                 tokenizer_path: Optional[str]=None) -> None:
        """
        """
        self.window_size = window_size
        self.uniform_count = uniform_count

        # if tokenizer is not provided, falls back to the default
        # tokenizer, which trained on the English Wikipedia dump (dumped at 2021/03)
        self._tokenizer = init_tokenizer(tokenizer_path)

    @property
    def n_tokens(self) -> int:
        return self._tokenizer.get_vocab_size()

    def build_matrix(self,
                     lines: Union[list[str], TextIO],
                     symmetrization: bool=True) -> sp.coo_matrix:
        """
        both file pointer for the text file and pre-load list of strings
        can be fed to this function
        """
        cooccur = dict()
        for line in lines:
            # strip line break
            if isinstance(line, str):
                line = line.replace('\n', '')

            # tokenize
            token_ids = tokenize(self._tokenizer, line)
            update_cooccurance(token_ids, cooccur, self.window_size)

        # convet cooccur matrix to coo matrix
        self.mat = cooccur2spmat(cooccur, self.n_tokens, symmetrization)
        return self.mat

    def from_file(self, corpus_fn: str) -> None:
        """ load the dumped (processed) corpus file to the memory
        """
        with open(corpus_fn, 'rb') as fp:
            data = pkl.load(fp)

        self._tokenizer = init_tokenizer(dump_str=data['tokenizer'])
        self.mat = sp.coo_matrix(
            (data['mat']['counts'], (data['mat']['row'], data['mat']['col'])),
            shape = (self.n_tokens, self.n_tokens)
        )
        self.window_size = data['window_size']
        self.uniform_count = data['uniform_count']

    def save(self, out_fn: str) -> None:
        """ save the processed corpus file to the disk
        """
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


def tokenize(tokenizer: Tokenizer,
             line: str,
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
            output.extend(list(tok.ids))
        return output
    else:
        tok = tokenizer.encode(line)
        return tok.ids


def update_cooccurance(token_ids: list[int],
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
                cooccur[row] = dict()

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
    mat = sp.coo_matrix((V, (I, J)), shape=(n_tokens, n_tokens))

    # to make sure the symmetry
    if symmetrization:
        mat_t = mat.T
        mat_t.setdiag(0)  # to avoid self-loop added twice
        mat_t.eliminate_zeros()
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
