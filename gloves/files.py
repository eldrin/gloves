import os
from os.path import join
import pkg_resources


TOKENIZER_FN = 'tokenizer-wiki210320-mxm.json'


def pretrained_tokenizer():
    """ read the filename of pre-trained tokenizer
    """
    return pkg_resources.resource_filename(
        __name__, join('data', TOKENIZER_FN)
    )


def faruqui_wordsim_evalset_url():
    """ release file of Faruqui's word-similarity evaluation set
    """
    return 'https://api.github.com/repos/mfaruqui/eval-word-vectors/tarball/v1.0'
