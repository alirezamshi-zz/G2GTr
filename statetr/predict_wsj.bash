#!/bin/bash
model_name=""
datapath=""
batch_size=40
main_path=""
output_path=""

if [ ! -d $output_path ]; then
  mkdir -p $output_path;
fi


python $main_path/test.py --batchsize $batch_size --model_name $model_name --datapath $output_path --testfile $datapath --mainpath $main_path
python $main_path/dep2conllx.py $datapath $main_path $model_name > $output_path/pred.conllx
perl $main_path/eval.pl -g $datapath -s $output_path/pred.conllx -q
