#!/bin/bash
MAIN_PATH=""
BERT_PATH=""
DATA_PATH=""

python run.py train --lr1 1e-5 --lr2 1e-4 -w 0.001 --modelname senttr_g2g --batch_size 40 --buckets 10 \
         --ftrain $DATA_PATH/train.conll \
         --ftrain_seq $DATA_PATH/train.seq \
         --ftest $DATA_PATH/test.conll \
         --fdev $DATA_PATH/dev.conll \
         --bert_path $BERT_PATH --punct --n_attention_layer 6 --epochs 12 \
         --input_graph --act_thr 210 --use_two_opts --main_path $MAIN_PATH
