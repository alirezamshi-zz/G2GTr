#!/bin/bash
model_name=""
datapath=""
batch_size=16
main_path=""
split="test"
lang=""
output_path=""

if [ ! -d $output_path ]; then
  mkdir -p $output_path;
fi


perl $main_path/conllu_to_conllx.pl < $datapath/$lang-ud-$split.conllu > $output_path/org.conllx
perl $main_path/conllu_to_conllx_no_underline.pl < $datapath/$lang-ud-$split.conllu > $output_path/original_nounderline.conllx

python $main_path/test.py --batchsize $batch_size --model_name $model_name --model_name2 $model_name --datapath $output_path --testfile org.conllx
python $main_path/dep2conllx.py $output_path/org.conllx $model_name > $output_path/pred.conllx

python $main_path/substitue_underline.py $output_path/original_nounderline.conllx $output_path/pred.conllx $output_path/pred_nounderline.conllx
perl $main_path/restore_conllu_lines.pl $output_path/pred_nounderline.conllx $datapath/$lang-ud-$split.conllu  > $output_path/pred.conllu

python $main_path/ud_eval.py $datapath/$lang-ud-$split.conllu $output_path/pred.conllu -v
