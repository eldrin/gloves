import sys
from typing import Optional
from os.path import exists, splitext, basename, join
import logging
import glob
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import hashlib
from io import BytesIO
from pathlib import Path
import subprocess

import numpy as np
from scipy import sparse as sp

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel

from .files import (pretrained_tokenizer,
                    faruqui_wordsim_evalset_url,
                    text8_url)
# from .evaluation import EvaluationSet


logger = logging.getLogger(__name__ + '.utils')


GLOVES_DATA_HOME = Path.home() / '.gloves'


def count_lines(textfile: str) -> int:
    """ count lines of textfile.

    currently, faster operation only works with *nix machines where
    `wc` is installed
    """
    if sys.platform.startswith('win'):
        logger.warning('[Warning] on windows, counting line number '
                       'is likely slow...')
        num_line = 0
        with open(textfile) as f:
            for line in f:
                num_line += 1
    else:
        num_line = int(
            subprocess
            .check_output(['wc', '-l', textfile])
            .decode("utf8")
            .split()[0]
        )
    return num_line


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


def argpart_sort(s, k, ascending=True):
    if ascending:
        p = s
    else:
        p = -s
    idx = np.argpartition(p, kth=k)[:k]
    return idx[np.argsort(p[idx])]


def download_faruqui(url: str=faruqui_wordsim_evalset_url()):
    """ download Faruqui dataset if not yet exists
    """
    import tarfile

    # get home directory and make `.gloves` directory
    data_path = GLOVES_DATA_HOME / 'faruqui_wordsim_evalset'
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    # download the tarball file in the folder
    logger.info("Fetching Faruqui's word-sim evlauation set..")
    try:
        response = urlopen(url)
    except HTTPError as e:
        raise SystemExit(e)
    except URLError as e:
        raise SystemExit(e)

    # if everything seems fine, extract to get the word-sim evaluation data
    logger.info("Writing the fetched results to disk..")
    with tarfile.open(fileobj=BytesIO(response.read())) as tarf:
        # extract word-sim rating files and save to the disk
        for member in tarf.getmembers():
            if 'EN-' in member.name:
                tarf.extract(member, data_path.as_posix())

    # close the urlopen
    response.close()


def load_faruqui_wordsim_evalset():
    """ Load Faruqui's collection of Word-similarity ratings
    """
    data_path = GLOVES_DATA_HOME / 'faruqui_wordsim_evalset'
    if not data_path.exists():
        download_faruqui()

    file_path = data_path / 'mfaruqui-eval-word-vectors-74458ad' / 'data' / 'word-sim'
    valid_fns = glob.glob(join(file_path.as_posix(), '*.txt'))
    if len(valid_fns) == 0:
        raise ValueError('[ERROR] no validation dataset found!')

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


def check_md5_text8(text8zip_path):
    """
    """
    # text8_md5 = '3bea1919949baf155f99411df5fada7e'  # from Mahoney's website
    text8_md5 = 'f26f94c5209bc6159618bad4a559ff81'  # one above SEEMS wrong, so I'm using this one
                                                    # from current downloadable file
    this_md5 = hashlib.md5(text8zip_path.open('rb').read()).hexdigest()

    if this_md5 != text8_md5:
        raise ValueError("[ERROR] MD5 checksum doesn't match!")

    return True  # it's good


def download_text8(url: str=text8_url()):
    """ download Text8 dataset if not yet exists

    TODOs:
      [ ] nice progress bar
      [ ] faster host?
    """
    import zipfile

    # get home directory and make `.gloves` directory
    data_path = GLOVES_DATA_HOME / 'text8'
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    file_path = data_path / 'text8.zip'

    if file_path.exists() and check_md5_text8(file_path):
        logger.info("We already have the file. aborting...")
        return

    # download the tarball file in the folder
    logger.info("Fetching Text8 corpus..")
    try:
        response = urlopen(url)
    except HTTPError as e:
        raise SystemExit(e)
    except URLError as e:
        raise SystemExit(e)

    logger.info("Writing the fetched results to disk..")
    # we will check MD5 the file first, and then extract
    with open(file_path, 'wb') as f:
        f.write(response.read())

    # check file health
    check_md5_text8(file_path)

    # if it's good, extract it
    with zipfile.ZipFile(file_path) as zipf:
        zipf.extract('text8', data_path.as_posix())

    # close the urlopen
    response.close()


def check_exists(token: str,
                 token_inv_map: dict[str, int],
                 tokenizer: Optional[Tokenizer] = None) -> Optional[int]:
    """
    """
    # here we check if the whole form of token is in tokenizer
    # otherwise, it'll give split list of sub-tokens
    if tokenizer is not None:
        tok = tokenizer.encode(token)
        if len(tok.tokens) == 1:
            return tok.ids[0]
        else:
            return None

    # if tokenizer is not given, use the token inverse map
    if token in token_inv_map:
        return token_inv_map[token]

    # otherwise, return None, as indication of `not-found`
    return None
