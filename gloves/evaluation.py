from typing import Optional, Union, TypedDict, Callable
import numpy as np
from scipy import sparse as sp
from scipy.stats import kendalltau, spearmanr

from tokenizers import Tokenizer

from .model import GloVe
from .corpus import Corpus
from .utils import check_exists


# alias for the score object
class EvaluationScore(TypedDict):
    corr: float    # correlation value
    p: float       # corresponding p-value
    nan_rate:float # number of NaN cases (i.e., due to missing word)

# alias for the other data objects
FaruquiEvalSet = dict[str, dict[frozenset, float]]
EvaluationSet = Union[FaruquiEvalSet, sp.coo_matrix]
EvaluationResult = dict[str, EvaluationScore]
Predictions = dict[str, tuple[str, str]]


def split_data(coo, train_ratio=0.8, valid_ratio=0.5, to_csr=True):
    """
    """
    rnd_idx = np.random.permutation(coo.nnz)
    n_train = int(len(rnd_idx) * train_ratio)
    n_valid = int(len(rnd_idx) * (1 - train_ratio) * valid_ratio)

    trn_idx = rnd_idx[:n_train]
    vld_idx = rnd_idx[n_train:n_train + n_valid]
    tst_idx = rnd_idx[n_train + n_valid:]

    outputs = tuple(
        sp.coo_matrix(
            (coo.data[idx], (coo.row[idx], coo.col[idx])),
            shape=coo.shape
        )
        for idx in [trn_idx, vld_idx, tst_idx]
    )

    if to_csr:
        return tuple(x.tocsr() for x in outputs)
    else:
        return outputs


def compute_similarities(glove: GloVe,
                         tokenizer: Tokenizer,
                         eval_set: EvaluationSet,
                         token_inv_map: dict[str, int],
                         score_method: str='cosine') -> Predictions:
    """
    """
    eps = 1e-20

    W = glove.embeddings_['W']

    # normalized embeddings for computing cosine distance easily
    if score_method == 'cosine':
        W = W / (np.linalg.norm(W, axis=1)[:, None] + eps)

    predictions = {}
    for dataset, ratings in eval_set.items():
        predictions[dataset] = {}

        for pair, rating in ratings.items():
            if len(pair) == 1:
                w1 = w2 = next(iter(pair))
            else:
                w1, w2 = pair

            # can't estimate the similarity due to the coverage
            i1 = check_exists(w1, token_inv_map, tokenizer)
            i2 = check_exists(w2, token_inv_map, tokenizer)
            if i1 is None or i2 is None:
                predictions[dataset][pair] = None
                continue

            predictions[dataset][pair] = W[i1] @ W[i2]

    return predictions


def compute_scores(eval_set: FaruquiEvalSet,
                   predictions: Predictions,
                   corr_func: Callable=spearmanr) -> EvaluationResult:
    """
    """
    scores = {}
    for dataset, pred in predictions.items():
        judges = eval_set[dataset]

        # get the prediction / jugement pairs
        pred_judge = [(judges[k], v) for k, v in pred.items()]
        p, j = list(zip(*[x for x in pred_judge if x[1] is not None]))

        # compute missing rate
        n_nans = len(pred) - len(j)
        nan_rate = n_nans / len(pred)

        # compute (non-parametric) correlation
        corr = corr_func(p, j)

        # register the row
        scores[dataset] = {'corr': corr.correlation,
                           'p': corr.pvalue,
                           'nan_rate': nan_rate}
    return scores
