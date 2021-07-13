# Graph-to-Graph Transformer for Transition-based Dependency Parsing (Sentence Transformer)

Pytorch implementation of the paper for the Sentence Transformer model

## Dependencies : 
You should install the following packages for train/testing the model: 
- Python 3.7
- [Pytorch](https://pytorch.org/) > 1.4.0 
- [Numpy](https://numpy.org/)
- [transformers](https://github.com/huggingface/transformers)
- [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
- [Torchvision](https://pytorch.org/)

Or the easiest way is to run the following command:  
```
conda env create -n senttr -f environment.yml
```

## Preparing the data:
For each dataset, we should do some pre-processing steps to build the proper input.
### WSJ Penn Treebank:
Download the data from [here](https://catalog.ldc.upenn.edu/LDC99T42). 
Now, convert constituency format to Stanford dependency style by following 
[this repository](https://github.com/hankcs/TreebankPreprocessing).  
Now, you can build the gold oracle for training data as follows (it's based on [arc-swift](https://github.com/qipeng/arc-swift) repo):  

```
cd preprocess/utils
./create_mappings.sh ../data/train.conll > mappings-ptb.txt
cd preprocess/src
python gen_oracle_seq.py ../data/train.conll train.seq --transsys ASd --mappings ./utils/mappings-ptb.txt
```
To include `SWAP` operation, you should update `transition.py` and `parserstate.py` files of arc-swift repository with our `transition.py` and `parserstate.py` files.  

Finally, you should replace the gold PoS tags with the predicted ones from [Stanford PoS tagger](https://nlp.stanford.edu/software/tagger.shtml).
You can use [this repository](https://github.com/shuoyangd/hoolock) to do this replacement.

### UD Treebanks:

Download the data from [here](http://hdl.handle.net/11234/1-2895). 
Since our models work with CoNLL-X format, you should convert dataset from CoNLL-U format to CoNLL-X format with [this tool](https://github.com/alirezamshi/G2GTr/blob/master/senttr/conllu_to_conllx_no_underline.pl). Then, you can find oracles by the modified version of arc-swift, as mentioned in above section.
## Training :

To train Sentence Transformer model, and its combination with Graph2Graph Transformer, you can check the following details:  

```
run.py train [-h] [--buckets BUCKETS] [--epochs EPOCHS] [--punct]
                    [--ftrain FTRAIN] [--ftrain_seq FTRAIN_SEQ] [--fdev FDEV]
                    [--ftest FTEST] [--warmupproportion WARMUPPROPORTION]
                    [--lowercase] [--lower_for_nonbert]
                    [--modelname MODELNAME] [--lr LR] [--lr2 LR2]
                    [--input_graph] [--layernorm_key] [--layernorm_value]
                    [--use_two_opts] [--mlp_dropout MLP_DROPOUT]
                    [--weight_decay WEIGHT_DECAY]
                    [--max_grad_norm MAX_GRAD_NORM] [--max_seq MAX_SEQ]
                    [--n_attention_layer N_ATTENTION_LAYER] [--checkpoint]
                    [--act_thr ACT_THR] [--bert_path BERT_PATH]
                    [--main_path MAIN_PATH] [--conf CONF] [--model MODEL]
                    [--vocab VOCAB] [--device DEVICE] [--seed SEED]
                    [--threads THREADS] [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --buckets BUCKETS     Max number of buckets to use
  --epochs EPOCHS       Number of training epochs
  --punct               Whether to include punctuation
  --ftrain FTRAIN       Path to train data
  --ftrain_seq FTRAIN_SEQ
                        Path to train oracle file
  --fdev FDEV           Path to dev file
  --ftest FTEST         Path to test file
  --warmupproportion WARMUPPROPORTION, -w WARMUPPROPORTION
                        Warm up proportion for BertAdam optimizer
  --lowercase           Whether to do lowercase in tokenisation step
  --lower_for_nonbert   Divide warm-up proportion of optimiser for randomly
                        initialised parameters
  --modelname MODELNAME
                        Path to saved checkpoint
  --lr LR               Learning rate for optimizer (for BERT parameters if
                        two optimisers used)
  --lr2 LR2             Learning rate for non-BERT parameters (two optimisers)
  --input_graph         Input dependency graph to attention mechanism
  --layernorm_key       layer normalization for Key (graph input)
  --layernorm_value     layer normalization for Value (graph input)
  --use_two_opts        Use one optimizer for Bert and one for others
  --mlp_dropout MLP_DROPOUT
                        MLP drop out
  --weight_decay WEIGHT_DECAY
                        Weight Decay
  --max_grad_norm MAX_GRAD_NORM
                        Clip gradient
  --max_seq MAX_SEQ     Maximum number of actions per sentence
  --n_attention_layer N_ATTENTION_LAYER
                        Number of Attention Layers
  --checkpoint          Start from a checkpoint
  --act_thr ACT_THR     Maximum number of actions per sentence (training data)
  --bert_path BERT_PATH
                        path to BERT
  --main_path MAIN_PATH
                        path to main directory
  --conf CONF, -c CONF  path to config file
  --model MODEL, -m MODEL
                        path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocab file
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --batch_size BATCH_SIZE
                        max num of buckets to use
```

To replicate our results, you can run ```senttr.bash``` for the baseline, and ```senttr_g2g.bash``` for the integrated model.  

## Evaluation:

To evaluate the model, you can check the following input requirements:  

```
run.py predict [-h] [--fdata FDATA] [--fpred FPRED]
                      [--modelname MODELNAME] [--mainpath MAINPATH]
                      [--conf CONF] [--model MODEL] [--vocab VOCAB]
                      [--device DEVICE] [--seed SEED] [--threads THREADS]
                      [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --fdata FDATA         Path to test dataset
  --fpred FPRED         Prediction path
  --modelname MODELNAME
                        Path to trained model
  --mainpath MAINPATH   Main path
  --conf CONF, -c CONF  path to config file
  --model MODEL, -m MODEL
                        path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocab file
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --batch_size BATCH_SIZE
                        max num of buckets to use
```

To predict and evaluate your trained model, fill requirements (data path, prediction path, model path) in the ```predict.bash``` file, then it
produces the predicted output CoNLL file, LAS, and UAS scores.  
