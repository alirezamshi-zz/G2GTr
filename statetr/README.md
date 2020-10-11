# Graph-to-Graph Transformer for Transition-based Dependency Parsing (State Transformer)

Pytorch implementation of the paper for the State Transformer model  

## Dependencies: 
You should install the following packages for training/evaluating the model: 
- Python 3.6
- [Pytorch](https://pytorch.org/) > 1.0.1 
- [Numpy](https://numpy.org/)
- [transformers](https://github.com/huggingface/transformers)
- [pytorch-pretrained-bert](https://github.com/huggingface/transformers/tree/v0.6.2)
- [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
- [Torchvision](https://pytorch.org/)

Or the easiest way is to run the following command:  
```
conda env create -n statetr -f statetr.yml
```  

## Preparing the data:  
Just follow the exact instruction that is described in "Sentence Transformer" repository.  

## Training:

Here are all paramters that are needed to train your own model:  

```
usage: run.py [-h] [--withpunct] [--graphinput] [--poolinghid] [--unlabeled]
              [--freezedp] [--lowercase] [--usepos] [--Bertoptim]
              [--pretrained] [--withbert] [--bertname BERTNAME]
              [--bertpath BERTPATH] [--fhistmodel] [--fcompmodel]
              [--layernorm] [--multigpu] [--seppoint] [--mean_seq]
              [--language LANGUAGE] [--datapath DATAPATH]
              [--trainfile TRAINFILE] [--devfile DEVFILE]
              [--testfile TESTFILE] [--seqpath SEQPATH]
              [--outputname OUTPUTNAME] [--batchsize BATCHSIZE]
              [--nepochs NEPOCHS] [--real_epoch REAL_EPOCH] [--lr LR]
              [--shuffle] [--ffhidden FFHIDDEN] [--clipword CLIPWORD]
              [--nclass NCLASS] [--ffdropout FFDROPOUT]
              [--nlayershistory NLAYERSHISTORY] [--embsize EMBSIZE]
              [--maxsteplength MAXSTEPLENGTH] [--updatelr UPDATELR]
              [--hiddensizelabel HIDDENSIZELABEL] [--histsize HISTSIZE]
              [--labelemb LABELEMB] [--nattentionlayer NATTENTIONLAYER]
              [--nattentionheads NATTENTIONHEADS]
              [--warmupproportion WARMUPPROPORTION] [--modelpath MODELPATH]
              [--use_topbuffer] [--use_justexist] [--use_two_opts]
              [--lr_nonbert LR_NONBERT] [--mainpath MAINPATH]

optional arguments:
  -h, --help            show this help message and exit
  --withpunct           Use punctuation
  --graphinput          Input graph to the model
  --poolinghid          Max Pooling the last hidden layer instead of CLS
  --unlabeled           Unlabeled dependency parsing
  --freezedp            Freeze the dependency relation embeddings
  --lowercase           Lowercase the words
  --usepos              Use POS tagger
  --Bertoptim           Use BertAdam for optimization
  --pretrained          Start with a checkpoint
  --withbert            Initialize the model with BERT
  --bertname BERTNAME   Type of Pre-trained BERT
  --bertpath BERTPATH   Type of Pre-trained BERT
  --fhistmodel          Apply history model
  --fcompmodel          Apply composition model
  --layernorm           Layer normalization for graph input
  --multigpu            Run the model on multiple GPUs
  --seppoint            Use CLS for dependency classifiers or graph output
                        mechanism
  --mean_seq            Used for computing total number of steps
  --language LANGUAGE   Language to train
  --datapath DATAPATH   Data directory for train/test
  --trainfile TRAINFILE
                        File to train the model
  --devfile DEVFILE     File to validate the model
  --testfile TESTFILE   File to test the model
  --seqpath SEQPATH     File to test the model
  --outputname OUTPUTNAME
                        Name of the output model
  --batchsize BATCHSIZE
                        Batch size number
  --nepochs NEPOCHS     Number of epochs
  --real_epoch REAL_EPOCH
                        Number of epochs that is reduced from total epochs
                        (checkpoint)
  --lr LR               Learning rate for training
  --shuffle             Shuffle training inputs
  --ffhidden FFHIDDEN   Size of hidden layer in classifier
  --clipword CLIPWORD   Percentage of keeping the orginal words of dataset
  --nclass NCLASS       Number of classes in classifier
  --ffdropout FFDROPOUT
                        Amount of drop-out in classifier
  --nlayershistory NLAYERSHISTORY
                        Number of layers in LSTM history model
  --embsize EMBSIZE     Dimension of Embeddings
  --maxsteplength MAXSTEPLENGTH
                        Maximum size of steps to de done on validation/test
                        time
  --updatelr UPDATELR   Step to update the learning rate
  --hiddensizelabel HIDDENSIZELABEL
                        Size of hidden layer in label classifier
  --histsize HISTSIZE   Size of embedding in history model
  --labelemb LABELEMB   Size of label embeddings
  --nattentionlayer NATTENTIONLAYER
                        Number of layers in self-attention model
  --nattentionheads NATTENTIONHEADS
                        Number of attention heads in self-attention model
  --warmupproportion WARMUPPROPORTION
                        Proportion of warm-up for BertAdam optimizer
  --modelpath MODELPATH
                        Name of the pretrained model
  --use_topbuffer       Use also top element of Buffer
  --use_justexist       Use top buffer just for exist classifier
  --use_two_opts        Use two optimizers for training
  --lr_nonbert LR_NONBERT
                        Learning rate for non-bert
  --mainpath MAINPATH   File to pre-trained char embeddings
```

To reproduce results of the paper, you can run the model with ```statetr.bash``` for baseline model, and ```statetr_g2g.bash``` 
for the integrated one.  

## Evaluation:

To evaluate a trained model, add the location of saved model, the input file, and output path to ```predict.bash``` file, 
then it computes the output CoNLL file, and LAS (UAS) scores.  
Here are the input requirements for evaluation:  

```
usage: test.py [-h] [--datapath DATAPATH] [--testfile TESTFILE]
               [--model_name MODEL_NAME] [--batchsize BATCHSIZE]
               [--mainpath MAINPATH]

optional arguments:
  -h, --help            show this help message and exit
  --datapath DATAPATH   Data directory for train/test
  --testfile TESTFILE   File to test the model
  --model_name MODEL_NAME
                        Model directory
  --batchsize BATCHSIZE
                        Batch size number
  --mainpath MAINPATH   File to test the model
```

```eval.pl``` file is used as the official evaluation script for English Penn Treebank.

## Error Analysis:  

To replicate Figure 3 and Table 3 of the paper, you can donwload [MaltEval tool](https://cl.lingfil.uu.se/~nivre/research/MaltEval.html), and use
the output predictions of ```predict.bash``` file, and gold dependencies to re-create plots. It's so easy, just one command!  

## TODO:

We will release pre-trained models at the conference time.
