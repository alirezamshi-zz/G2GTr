#!/bin/bash
main_path="/idiap/temp/amohammadshahi/Debug_transformer/emnlp/EMNLP2020/emnlp-senttr"
input_data="/idiap/temp/amohammadshahi/Debug_transformer/edited-transformer-new-ud-swap/data/test.conll"
type="conllx"
modelname="senttr"
output_path="idiap/temp/amohammadshahi/Debug_transformer/emnlp/EMNLP2020/emnlp-senttr/predict_output"

if [ ! -d $output_path ]; then
  mkdir -p $output_path;
fi
if [ "$type" = "conllu" ]; then
    echo "Input is CONLL-U format"
    perl $main_path/conllu_to_conllx.pl < $input_data > $output_path/original.conllx
    perl $main_path/conllu_to_conllx_no_underline.pl < $input_data > $output_path/original_nounderline.conllx
else
    echo "Input is CONLL-X format"
    cp $input_data $output_path/original.conllx
fi

echo "Predicting the input file"
python run.py predict --modelname $modelname --fdata $output_path/original.conllx --fpred $output_path/pred.conllx --mainpath $main_path/
echo "Finished Prediction"
if [ "$type" = "conllu" ]; then
    echo "Converting back to CONLL-U format"
    python substitue_underline.py $output_path/original_nounderline.conllx $output_path/pred.conllx $output_path/pred_nounderline.conllx
    perl $main_path/restore_conllu_lines.pl $output_path/pred_nounderline.conllx $input_data  > $output_path/pred.conllu
else
    echo "Output is CONLL-X format"
fi

if [ "$type" = "conllu" ]; then
    echo "Evaluating based on official UD script"
    python $main_path/ud_eval.py $input_data $output_path/pred.conllu -v
else
    if [ "$type" = "conllx" ]; then
        perl eval.pl -g $output_path/original.conllx -s $output_path/pred.conllx -q
    else
        perl eval.pl -g $output_path/original.conllx -s $output_path/pred.conllx -q -p
    fi
    echo "done"
fi
