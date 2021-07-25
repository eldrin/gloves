GloVe-ALS
=========

Implementation of [GloVe](https://github.com/stanfordnlp/GloVe) using [alternating least squares (ALS)](https://ieeexplore.ieee.org/abstract/document/4781121?casa_token=3EiDn2ITeiAAAAAA:JlpN1YDtJjwGNFcztjWdIBbcJCNlnPcPoH7OcStUJFbe8T-NXU_mqPTPvom-vfFD5pPn8s5m) algorithm, specifically using the element-wise update trick by [Pilászy et al.](https://dl.acm.org/doi/10.1145/1864708.1864726). The aim for this implementation is reasonably fast and robust training.


## Get Started

Currently, installing via `github` and `pip` is the only way at the moment.

```bash
pip install git+https://github.com/eldrin/gloveals.git@master
```

## Minimal Example

```python
from gloveals.models import GloVeALS
from gloveals.evaluation import split_data

# X is scipy sparse matrix with the shape of
# (|target words|, |context words|). `split_data` splits the
# non-zero elements of the X into train / valid / test set
# with a certain proportions (default 0.8 / 0.1 / 0.1)
Xtr, Xvl, Xts = split_data(X.tocoo())

# GloveALS is scikit-learn-like object
glove = GloVeALS(
    n_components=32,
    l2=1,
    init=1e-3,
    n_iters=15,
    alpha=3/4.,
    x_max=100,
    dtype=np.float32
)
glove.fit(Xtr, verbose=True)

mse_valid = glove.score(Xvl, weighted=False)

print(f'Validation Loss (without weight): {mse_valid:.4f}')
```


## What's the benefit of ALS?

Unlike the (stochastic) gradient descent method, ALS generally converges in a couple of dozens of epochs of updates, while less hyper-parameters to be tuned [^1]. Also, it employs the sparsity efficiently such that the computation for one epoch only costs $O(K|R|)$ where $K$ denotes the dimensionality of the low-rank approximation and $|R|$ is the number of non-zero element in given matrix. For the word co-occurance matrix, sparsity usually reaches (roughly) from 75% to 95%, it is desirable for the speedup.

[^1]: Specifically, it does not require hyper parameters regarding the gradient descent update (i.e., learning rate, momentum related ones if one uses algorithms such as Adagrad or ADAM).


## TODO

- [ ] writing test
- [ ] adding feature
  - [ ] `.most_similar` method
  - [ ] utility to convert embedding to `gensim` model?
- [ ] benchmark (vs. [`glove-python`](https://github.com/maciejkula/glove-python))


## Authors

Jaehun Kim


## License

This project is licensed under the MIT License - see the LICENSE.md file for details

<!-- ## Acknowledgement -->
