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
