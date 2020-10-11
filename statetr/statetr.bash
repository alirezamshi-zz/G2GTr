#!/bin/bash
DATA_PATH="/idiap/temp/amohammadshahi/Debug_transformer/edited-transformer-new-ud-swap/data"
BERT_NAME="bert-base-uncased"
BERT_PATH="/idiap/temp/amohammadshahi/Debug_transformer/graph-based-g2g-parser/"
MAIN_PATH="/idiap/temp/amohammadshahi/Debug_transformer/emnlp/EMNLP2020/emnlp_statetr"
python  run.py --mean_seq --lr 1e-5 --lowercase --usepos --withpunct \
        --batchsize 40 --nepochs 12 --warmupproportion 0.01 --Bertoptim \
        --nattentionlayer 6 --nattentionheads 12 --seppoint --withbert --fhistmodel --fcompmodel \
        --outputname statetr --mainpath $MAIN_PATH \
        --datapath $DATA_PATH \
        --bertname $BERT_NAME --bertpath $BERT_PATH --use_topbuffer --use_justexist
