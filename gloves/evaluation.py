import numpy as np
from scipy import sparse as sp


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