#!/bin/bash

# change this for your own choice
DATA_PATH="${HOME}"

# download text8. it takes a while
python -c "from gloves import utils; utils.download_text8()"

# process the merged file with `cooccur` tool wrt window sizes
for ws in 3 5 7 9 11 13
do
    echo "processing fold: ${i} - window size: ${ws}..."
    # process
    cooccur build \
        -p $DATA_PATH \
        -w $ws \
        --symmetrization --weighted-count --quiet \
        "${DATA_PATH}/.gloves/text8/text8"

    # rename the file not to be overwritten
    mv text8.cooccur.pkl "text8.ws${ws}.cooccur.pkl"
done

# search the best model
gloves optimize \
    -o final_model.glv.mdl \
    --data-filename-template 'text8.ws{window_size:d}.cooccur.pkl' \
    --n-calls 50 \
    --eval-set 'faruqui' \
    $DATA_PATH $DATA_PATH

