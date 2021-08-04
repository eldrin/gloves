import os
from os.path import join
import pkg_resources


TOKENIZER_FN = 'tokenizer-wiki210320-mxm.json'
DEFAULT_OPTHYPER_FN = 'default_search_option.json'


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


def default_optimize_config():
    """ read the filename of default `optimize` config file
    """
    return pkg_resources.resource_filename(
        __name__, join('data', DEFAULT_OPTHYPER_FN)
    )


def text8_url():
    """ text8 download URL

    directly from Matt Mahoney's website (http://mattmahoney.net/dc/textdata)

    TODO: check the license stuffs
    """
    return 'http://mattmahoney.net/dc/text8.zip'
