from typing import Optional
from os.path import exists, splitext, basename

from scipy import sparse as sp

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel

from .files import pretrained_tokenizer


def is_json(path: str) -> bool:
    """ simply checking the file extension is `json`
    """
    return splitext(basename(path))[-1] == '.json'


def is_valid(path: str) -> bool:
    """ check whether the tokenizer model file is valid

    TODO: currently it's just a file extension checker (for `.json`)
          should be elaborated to actually check the validity
    """
    return is_json(path)


def init_tokenizer(path: Optional[str]=None,
                   dump_str: Optional[str]=None) -> Tokenizer:
    """
    path: pre-trained (`hugginface`) tokenizer dump (`.json`)
    dump_str: stringified tokenizer model. this is prioritized
              if both arguments are given.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    if dump_str is not None:
        tokenizer = tokenizer.from_str(dump_str)
    else:
        if path is not None and (exists(path) and is_valid(path)):
            tokenizer = tokenizer.from_file(path)
        else:
            tokenizer = tokenizer.from_file(pretrained_tokenizer())
    return tokenizer


def is_symmetric(mat: sp.coo_matrix, tol=1e-10) -> bool:
    """ Copied directly from scikit-learn

    (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/validation.py)
    """
    diff = mat - mat.T
    if diff.format not in ['csr', 'csc', 'coo']:
        diff = diff.tocsr()
    symmetric = np.all(abs(diff.data) < tol)
    return symmetric
