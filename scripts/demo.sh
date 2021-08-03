#!/bin/bash

DATA_PATH="${HOME}"

# download text8. it takes a while
python -c "from gloves import utils; utils.download_text8()"

# make cooccurrence matrix (window_size:11)
cooccur build --no-quiet -p $DATA_PATH -w 11 "${HOME}/.gloves/text8/text8"

# train a glove model and save the result to the disk
gloves train \
    --n-components 256 \
    --solver 'sgd' \
    --n-iters 64 \
    --learning-rate 1e-1 \
    --no-quiet \
    --num-threads 8 \
    -p $DATA_PATH \
    "${DATA_PATH}/text8.cooccur.pkl"

# simple printout for the sanity check
python -c "from gloves.model import GloVe
glove = GloVe.from_file('${DATA_PATH}/model.glv.pkl')
print(glove.most_similar('text'))"
