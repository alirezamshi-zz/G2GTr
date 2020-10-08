# -*- coding: utf-8 -*-

import os
from os import path
from datetime import datetime, timedelta
from parser import Parser, Model
from parser.metric import Metric
from parser.utils import Corpus, Vocab
from parser.utils.data import TextDataset, batchify

import torch
from transformers import AdamW,get_linear_schedule_with_warmup
from parser.utils.corpus import read_seq

class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--buckets', default=64, type=int,
                               help='Max number of buckets to use')

        subparser.add_argument('--epochs', default=12, type=int,
                               help='Number of training epochs')

        subparser.add_argument('--punct', default=False, action='store_true',
                               help='Whether to include punctuation')

        subparser.add_argument('--ftrain', default='data/train.conll',
                               help='Path to train data')

        subparser.add_argument('--ftrain_seq', default='data/train.seq',
                               help='Path to train oracle file')

        subparser.add_argument('--fdev', default='data/dev.conll',
                               help='Path to dev file')

        subparser.add_argument('--ftest', default='data/test.conll',
                               help='Path to test file')

        subparser.add_argument('--warmupproportion', '-w', default=0.01, type=float,
                               help='Warm up proportion for BertAdam optimizer')

        subparser.add_argument('--lowercase', default=False, action='store_true',
                               help='Whether to do lowercase in tokenisation step')

        subparser.add_argument('--lower_for_nonbert', default=False, action='store_true',
                               help='Divide warm-up proportion of optimiser '
                                    'for randomly initialised parameters')

        subparser.add_argument('--modelname', default='None',
                               help='Path to saved checkpoint')

        subparser.add_argument('--lr', default=1e-5, type=float,
                               help='Learning rate for optimizer (for BERT parameters if two optimisers used)')

        subparser.add_argument('--lr2', default=2e-3, type=float,
                               help='Learning rate for non-BERT parameters (two optimisers)')

        subparser.add_argument('--input_graph', default=False, action='store_true',
                               help='Input dependency graph to attention mechanism')

        subparser.add_argument('--layernorm_key', default=False, action='store_true',
                               help='layer normalization for Key (graph input)')

        subparser.add_argument('--layernorm_value', default=False, action='store_true',
                               help='layer normalization for Value (graph input)')

        subparser.add_argument('--use_two_opts', default=False, action='store_true',
                               help='Use one optimizer for Bert and one for others')

        subparser.add_argument('--mlp_dropout', default=0.33,type=float,
                               help='MLP drop out')

        subparser.add_argument('--weight_decay', default=0.01,type=float,
                               help='Weight Decay')

        subparser.add_argument('--max_grad_norm', default=1.0,type=float,
                               help='Clip gradient')

        subparser.add_argument('--max_seq', default=1000,type=int,
                               help='Maximum number of actions per sentence')

        subparser.add_argument('--n_attention_layer', default=6,type=int,
                               help='Number of Attention Layers')

        subparser.add_argument('--checkpoint', default=False,action='store_true',
                               help='Start from a checkpoint')

        subparser.add_argument('--act_thr', default=210,type=int,
                               help='Maximum number of actions per sentence (training data)')

        subparser.add_argument('--bert_path', default='', help='path to BERT')

        subparser.add_argument('--main_path', default='', help='path to main directory')

        return subparser

    def __call__(self, config):
        print("Preprocess the data")
        train = Corpus.load(config.ftrain)
        dev = Corpus.load(config.fdev)
        test = Corpus.load(config.ftest)

        if path.exists(config.model) != True:
            os.mkdir(config.model)

        if path.exists("model/") != True:
            os.mkdir("model/")

        if path.exists(config.model+config.modelname) != True:
            os.mkdir(config.model+config.modelname)

        if config.checkpoint:
            vocab = torch.load(config.main_path + config.vocab+config.modelname + "/vocab.tag")
        else:
            vocab = Vocab.from_corpus(config=config, corpus=train,
                                      corpus_dev=dev,corpus_test=test,min_freq=0)
        train_seq = read_seq(config.ftrain_seq,vocab)
        total_act = 0
        for x in train_seq:
            total_act += len(x)
        print("number of transitions:{}".format(total_act))

        torch.save(vocab, config.vocab+config.modelname + "/vocab.tag")
        
        config.update({
            'n_words': vocab.n_train_words,
            'n_tags': vocab.n_tags,
            'n_rels': vocab.n_rels,
            'n_trans':vocab.n_trans,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })

        print("Load the dataset")
        trainset = TextDataset(vocab.numericalize(train,train_seq))
        devset = TextDataset(vocab.numericalize(dev))
        testset = TextDataset(vocab.numericalize(test))

        # set the data loaders
        train_loader,_ = batchify(dataset=trainset,
                                batch_size=config.batch_size,
                                n_buckets=config.buckets,
                                shuffle=True)
        dev_loader,_  = batchify(dataset=devset,
                              batch_size=config.batch_size,
                              n_buckets=config.buckets)
        test_loader,_ = batchify(dataset=testset,
                               batch_size=config.batch_size,
                               n_buckets=config.buckets)

        print(f"{'train:':6} {len(trainset):5} sentences in total, "
              f"{len(train_loader):3} batches provided")
        print(f"{'dev:':6} {len(devset):5} sentences in total, "
              f"{len(dev_loader):3} batches provided")
        print(f"{'test:':6} {len(testset):5} sentences in total, "
              f"{len(test_loader):3} batches provided")
        print("Create the model")

        if config.checkpoint:
            parser = Parser.load(config.main_path + config.model + config.modelname
                                 + "/parser-checkpoint")
        else:
            parser = Parser(config, vocab.bertmodel)

        print("number of parameters:{}".format(sum(p.numel() for p in parser.parameters()
                                             if p.requires_grad)))
        if torch.cuda.is_available():
            print('Train/Evaluate on GPU')
            device = torch.device('cuda')
            parser = parser.to(device)

        model = Model(vocab, parser, config, vocab.n_rels)
        total_time = timedelta()
        best_e, best_metric = 1, Metric()

        ## prepare optimisers
        num_train_optimization_steps = int(config.epochs * len(train_loader))
        warmup_steps = int(config.warmupproportion * num_train_optimization_steps)
        ## one for parsing parameters, one for BERT parameters
        if config.use_two_opts:
            model_nonbert = []
            model_bert = []
            layernorm_params = ['layernorm_key_layer', 'layernorm_value_layer',
                                'dp_relation_k', 'dp_relation_v']
            for name, param in parser.named_parameters():
                if 'bert' in name and not any(nd in name for nd in layernorm_params):
                    model_bert.append((name, param))
                else:
                    model_nonbert.append((name, param))

            # Prepare optimizer and schedule (linear warmup and decay) for Non-bert parameters
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters_nonbert = [
                {'params': [p for n, p in model_nonbert if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay},
                {'params': [p for n, p in model_nonbert if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            model.optimizer_nonbert = AdamW(optimizer_grouped_parameters_nonbert, lr=config.lr2)

            model.scheduler_nonbert = get_linear_schedule_with_warmup(model.optimizer_nonbert,
                                                                      num_warmup_steps=warmup_steps,
                                                                      num_training_steps=num_train_optimization_steps)

            # Prepare optimizer and schedule (linear warmup and decay) for Bert parameters
            optimizer_grouped_parameters_bert = [
                {'params': [p for n, p in model_bert if not any(nd in n for nd in no_decay)],
                    'weight_decay': config.weight_decay},
                {'params': [p for n, p in model_bert if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            model.optimizer_bert = AdamW(optimizer_grouped_parameters_bert, lr=config.lr)
            model.scheduler_bert = get_linear_schedule_with_warmup(
                model.optimizer_bert, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
            )

        else:
            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in parser.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': config.weight_decay},
                {'params': [p for n, p in parser.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            model.optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)
            model.scheduler = get_linear_schedule_with_warmup(
                model.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
            )


        start_epoch = 1

        ## load model, optimiser, and other parameters from a checkpoint
        if config.checkpoint:
            check_load = torch.load(config.main_path + config.model
                                    + config.modelname + "/checkpoint")
            if config.use_two_opts:
                model.optimizer_bert.load_state_dict(check_load['optimizer_bert'])
                model.optimizer_nonbert.load_state_dict(check_load['optimizer_nonbert'])
                model.scheduler_bert.load_state_dict(check_load['lr_schedule_bert'])
                model.scheduler_nonbert.load_state_dict(check_load['lr_schedule_nonbert'])
                start_epoch = check_load['epoch']+1
                best_e = check_load['best_e']
                best_metric = check_load['best_metric']
            else:
                model.optimizer.load_state_dict(check_load['optimizer'])
                model.scheduler.load_state_dict(check_load['lr_schedule'])
                start_epoch = check_load['epoch']+1
                best_e = check_load['best_e']
                best_metric = check_load['best_metric']

        f1 = open(config.model+config.modelname+"/baseline.txt","a")
        
        f1.write("New Model:\n")
        f1.close()
        for epoch in range(start_epoch, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            model.train(train_loader)
            print(f"Epoch {epoch} / {config.epochs}:")
            f1 = open(config.model+config.modelname+"/baseline.txt","a")
            dev_metric = model.evaluate(dev_loader, config.punct)
            f1.write(str(epoch)+"\n")
            print(f"{'dev:':6} {dev_metric}")
            f1.write(f"{'dev:':6} {dev_metric}")
            f1.write("\n")
            f1.close()

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                print(config.model + config.modelname + "/model_weights")
                model.parser.save(config.model + config.modelname + "/model_weights")
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break

            ## save checkpoint
            if config.use_two_opts:
                checkpoint = {
                    "epoch": epoch,
                    "optimizer_bert":model.optimizer_bert.state_dict(),
                    "lr_schedule_bert":model.scheduler_bert.state_dict(),
                    "lr_schedule_nonbert":model.scheduler_nonbert.state_dict(),
                    "optimizer_nonbert":model.optimizer_nonbert.state_dict(),
                    'best_metric':best_metric,
                    'best_e':best_e
                }
                torch.save(checkpoint,config.main_path + config.model + config.modelname + "/checkpoint")
                parser.save(config.main_path + config.model + config.modelname + "/parser-checkpoint")
            else:
                checkpoint = {
                    "epoch": epoch,
                    "optimizer":model.optimizer.state_dict(),
                    "lr_schedule":model.scheduler.state_dict(),
                    'best_metric':best_metric,
                    'best_e':best_e
                }
                torch.save(checkpoint,config.main_path + config.model + config.modelname + "/checkpoint")
                parser.save(config.main_path + config.model + config.modelname + "/parser-checkpoint")
        model.parser = Parser.load(config.model + config.modelname + "/model_weights")
        metric = model.evaluate(test_loader, config.punct)
        print(metric)
        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
