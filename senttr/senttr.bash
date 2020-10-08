#!/bin/bash
MAIN_PATH=""
BERT_PATH=""
DATA_PATH=""

python run.py train --lr 1e-5 -w 0.001 --modelname senttr --batch_size 40 --buckets 10 \
         --ftrain $DATA_PATH/train.conll \
         --ftrain_seq $DATA_PATH/train.seq \
         --ftest $DATA_PATH/test.conll \
         --fdev $DATA_PATH/dev.conll \
         --bert_path $BERT_PATH --punct --n_attention_layer 6 --epochs 12 --act_thr 280 \
         --main_path $MAIN_PATH
