#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import time
from collections import Counter
import torch.nn as nn
import numpy as np
import torch
import torch.cuda as cuda
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from model import BertConfig
from model import FFCompose
import pickle

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '[UNK]'
NULL = '[NULL]'
ROOT = '[ROOT]'
CLS = '[CLS]'
SEP = '[SEP]'

LEN_VOCAB = 0
class Parser(object):
    def __init__(self, dataset, opt, dataset_dev=None,dataset_test=None):
        
        self.embedding_shape = 0
        root_labels = list([
            l for ex in dataset for (h, l) in zip(ex['head'], ex['label'])
            if h == 0
        ])
        counter = Counter(root_labels)
        if len(counter) > 1:
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0]
        deprel = [self.root_label] + list(
            set([
                w for ex in dataset
                for w in ex['label'] if w != self.root_label
            ]))
        self.unlabeled = opt.unlabeled
        self.with_punct = opt.withpunct
        self.use_pos = opt.usepos
        self.language = opt.language
        
        if self.unlabeled:
            ## L:left-arc,R:right-arc,S:shift,H:swap
            transit = ['L', 'R', 'S','H']
            self.n_deprel = 1
        else:
            transit = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['UNK_LABEL']
            self.n_deprel = len(deprel)
            
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        
        self.n_transit = len(transit)
        self.L_NULL = len(transit)-1
        self.tran2id = {t: i for (i, t) in enumerate(transit)}
        self.id2tran = {i: t for (i, t) in enumerate(transit)}

        logging.info('Build dictionary for part-of-speech tags.')
        tok2id.update(
            build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                       offset=len(tok2id)))
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)
        
        
        logging.info('Build dictionary for words.')
        dataset_total = dataset + dataset_dev + dataset_test
        train_words = Counter([w for ex in dataset_total for w in ex['word']])
        clip = int(opt.clipword*len(train_words))
        train_words = train_words.most_common(clip)

        final_words = []
        for word in train_words:
            final_words.append(word[0])
        del train_words
        tok2id.update(
            build_dict(final_words,
                       offset=len(tok2id)))

        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)
        tok2id[CLS] = self.CLS = len(tok2id)
        tok2id[SEP] = self.SEP = len(tok2id)
        
        self.tok2id = tok2id
        
        self.id2tok = {v: k for (k, v) in tok2id.items()}

        self.n_tokens = len(tok2id)
        
        self.layers_lstm = opt.nlayershistory
        self.emb_size = opt.embsize

    def vectorize(self, examples):
        vec_examples = []
        for ex in examples:
            word = [self.ROOT] + [
                self.tok2id[w] if w in self.tok2id else self.UNK
                for w in ex['word']
            ]
            
            pos = [self.P_ROOT] + [
                self.tok2id[P_PREFIX + w]
                if P_PREFIX + w in self.tok2id else self.P_UNK
                for w in ex['pos']
            ]
            head = [-1] + ex['head']
            
            label = [-1] + [
                self.tok2id[L_PREFIX + w]
                if L_PREFIX + w in self.tok2id else -1 for w in ex['label']
            ]
            vec_examples.append({
                    'word': word,
                    'pos': pos,
                    'head': head,
                    'label': label
            })
        return vec_examples
        
    def create_instances(self, examples, seq_examples):
        all_instances = []
        for id, (ex,seq_ex) in enumerate(zip(examples,seq_examples)):
            n_words = len(ex['word']) - 1

            if 3 not in seq_ex[0]:
                if len(seq_ex[0]) == 2*n_words:
                    all_instances.append((ex,seq_ex[0],seq_ex[1]))
                else:
                    assert False,'wrong oracle!! word:{},oracle:{}'.format(ex,seq_ex[0])
            else:
                all_instances.append( (ex,seq_ex[0],seq_ex[1]) )
        return all_instances

    def legal_labels(self, len_stack, len_buffer, index_stack):
        labels = [1] if len_stack >= 2 and index_stack[-2] != 0 else [0]
        labels += [1] if len_stack >= 2 and index_stack[-1] != 0 else [0]
        labels += [1] if len_buffer > 0 else [0]
        labels += [1] if len_stack >= 2 and 0 < index_stack[-2] < index_stack[-1] else [0]
        return labels
    

## transit = ['L', 'R', 'S','H']
def read_seq(in_file, parser, reduced, thr):
    max_read = 0
    lines = []
    with open(in_file, 'r') as f:
        for line in f:
            lines.append(line)

    for i in range(len(lines)):
        lines[i] = lines[i].strip().split()

    gold_seq, arcs, seq = [], [], []
    for line in lines:

        if reduced and max_read == thr:
            break
        if len(line) == 0:
            gold_seq.append((seq, arcs))
            max_read += 1
            arcs, seq = [], []
        elif len(line) == 3:
            # print(line)
            assert line[0] == 'Shift'
            seq.append(2)
            arcs.append(parser.L_NULL)
        elif len(line) == 1:
            assert line[0] == 'Swap'
            seq.append(3)
            arcs.append(parser.L_NULL)
        elif len(line) == 2:
            if line[0].startswith('R'):
                assert line[0] == 'Right-Arc'
                seq.append(1)
                arcs.append(parser.tran2id['R-' + line[1]])
            elif line[0].startswith('L'):
                assert line[0] == 'Left-Arc'
                seq.append(0)
                arcs.append(parser.tran2id['L-' + line[1]])
    return gold_seq

    
# reading input data
def read_conll(in_file, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                examples.append({
                    'word': word,
                    'pos': pos,
                    'head': head,
                    'label': label
                })
                word, pos, head, label = [], [], [], []
                if (max_example is
                        not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({
                'word': word,
                'pos': pos,
                'head': head,
                'label': label
            })
            
    
    return examples


def build_dict(keys, n_max=None, offset=0):
    count = Counter()
    for key in keys:
        count[key] += 1

    if n_max is None:
        ls = count.most_common()
    else:
        ls = count.most_common(n_max)

    return {w[0]: index + offset for (index, w) in enumerate(ls)}

def punct(language, pos):
    if language == 'english':
        return pos in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    elif language == 'chinese':
        return pos == 'PU'
    elif language == 'french':
        return pos == 'PUNC'
    elif language == 'german':
        return pos in ["$.", "$,", "$["]
    elif language == 'spanish':
        # http://nlp.stanford.edu/software/spanish-faq.shtml
        return pos in [
            "f0", "faa", "fat", "fc", "fd", "fe", "fg", "fh", "fia", "fit",
            "fp", "fpa", "fpt", "fs", "ft", "fx", "fz"
        ]
    elif language == 'universal':
        return pos == 'PUNCT'
    else:
        raise ValueError('language: %s is not supported.' % language)

# preprocess the data without bert initialization
def filter_random(opt,parser):
    
    word_vectors = {}
    embeddings_matrix = np.asarray(
        np.random.normal(-0.0279, 0.041, (parser.n_tokens, opt.embsize)), dtype='float32')
    ## loading the pre-trained BERT model, then modifying the word embedding part
    tempbert = BertModel.from_pretrained(str(opt.bertpath)+str(opt.bertname))
    torch.save(tempbert.embeddings.position_embeddings.state_dict(),'position'+str(opt.outputname))
    torch.save(tempbert.embeddings.token_type_embeddings.state_dict(),'token_type'+str(opt.outputname))
    
    word_vectors = list(tempbert.parameters())[0].data.numpy()
    temptokenizer = BertTokenizer.from_pretrained(str(opt.bertpath)+str(opt.bertname))
    
    UNK_ID = temptokenizer.convert_tokens_to_ids(temptokenizer.tokenize('[UNK]'))
    
    for token in parser.tok2id:
        i = parser.tok2id[token]
        j = temptokenizer.convert_tokens_to_ids(temptokenizer.tokenize(token))
        if j != UNK_ID:
            embeddings_matrix[i] = word_vectors[j[0]]
            
    emb_size = embeddings_matrix.shape[1]
    embeddings_matrix = torch.from_numpy(embeddings_matrix)
    EMB = nn.Embedding(parser.n_tokens,emb_size,padding_idx=0)
    EMB.weight = nn.Parameter(embeddings_matrix)
    tempbert.embeddings.word_embeddings = EMB
    
    torch.save(tempbert.embeddings.word_embeddings.state_dict(),'word_emb'+str(opt.outputname))
    
    del EMB, tempbert, word_vectors, temptokenizer

    return embeddings_matrix

## preprocess the data with bert initialization
def filter_bert(opt,parser):
    
    word_vectors = {}
    embeddings_matrix = np.asarray(
        np.random.normal(-0.0279, 0.041, (parser.n_tokens, opt.embsize)), dtype='float32')
    ## loading the pre-trained BERT model, then modifying the word embedding part
    print(str(opt.bertpath)+str(opt.bertname))
    print("Use Normal BERT")
    tempbert = BertModel.from_pretrained(str(opt.bertpath)+str(opt.bertname))
    
    dict = tempbert.state_dict()
    keys = list(dict.keys())
    numbers = list(range(opt.nattentionlayer, 12))

    deleted = []
    for x in numbers:
        deleted.append(str(x))
    
    for dl in deleted:
        for key in keys:
            if key.find(dl) != -1:
                del dict[key]
    word_vectors = list(tempbert.parameters())[0].data.numpy()

    print("Use Normal BERT")
    temptokenizer = BertTokenizer.from_pretrained(str(opt.bertpath)+str(opt.bertname))
    
    UNK_ID = temptokenizer.convert_tokens_to_ids(temptokenizer.tokenize('[UNK]'))
    counter = 0.0
    total = 0.0
    for token in parser.tok2id:
        i = parser.tok2id[token]
        x = temptokenizer.tokenize(token)
        if "[UNK]" in x and len(x) == 1:
            counter+=1.0
        total +=1.0
        j = temptokenizer.convert_tokens_to_ids(temptokenizer.tokenize(token))

        if j != UNK_ID and len(j) > 0:
            embeddings_matrix[i] = word_vectors[j[0]]

    print("unk ratio: {}".format(counter/total*100))

    emb_size = embeddings_matrix.shape[1]
    embeddings_matrix = torch.from_numpy(embeddings_matrix)
    
    dict.update({'embeddings.word_embeddings.weight':embeddings_matrix})


    if opt.graphinput:
        del dict['embeddings.token_type_embeddings.weight']
    del dict['pooler.dense.weight']
    del dict['pooler.dense.bias']
    
    torch.save(dict,'small_bert'+str(opt.outputname))
    
    del tempbert, word_vectors, temptokenizer, dict
    
    return embeddings_matrix
    

# preprocess the data when starting from a checkpoint
def load_and_preprocess_datap(opt,parser,reduced=True):
    
    print("Loading data...", )
    start = time.time()
    train_set = read_conll(
        os.path.join(opt.datapath, opt.trainfile),
        lowercase=opt.lowercase)
    dev_set = read_conll(
        os.path.join(opt.datapath, opt.devfile),
        lowercase=opt.lowercase)
    test_set = read_conll(
        os.path.join(opt.datapath, opt.testfile),
        lowercase=opt.lowercase)
    
    thr = 128
    if reduced:
        train_set = train_set[:thr+3]
        dev_set = dev_set[:thr+1]
        test_set = test_set[:thr+1]
    
    print("took {:.2f} seconds".format(time.time() - start))

    print("Building parser...", )
    start = time.time()
    print("took {:.2f} seconds".format(time.time() - start))

    print("Reading gold actions...")
    seq_train = read_seq(os.path.join(opt.datapath, opt.seqpath), parser,reduced,thr+3)

    print("Loading pretrained embeddings...", )
    start = time.time()

    if opt.withbert:
        embeddings_matrix = filter_bert(opt,parser)
    else:
        embeddings_matrix = filter_random(opt,parser)

    print("took {:.2f} seconds".format(time.time() - start))

    print("Vectorizing data...", )
    start = time.time()
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Preprocessing training data...", )
    start = time.time()
    train_examples = parser.create_instances(train_set,seq_train)
    
    print("took {:.2f} seconds".format(time.time() - start))
    return embeddings_matrix, train_examples, train_set, dev_set, test_set, {'P':4}

# preprocess the data when not starting from a checkpoint
def load_and_preprocess_data(opt,reduced=True):
    
    print("Loading data...", )
    start = time.time()
    train_set = read_conll(
        os.path.join(opt.datapath, opt.trainfile),
        lowercase=opt.lowercase)
    dev_set = read_conll(
        os.path.join(opt.datapath, opt.devfile),
        lowercase=opt.lowercase)
    test_set = read_conll(
        os.path.join(opt.datapath, opt.testfile),
        lowercase=opt.lowercase)
    
    thr = 64
    if reduced:
        train_set = train_set[:thr+3]
        dev_set = dev_set[:thr+1]
        test_set = test_set[:thr+1]
    
    print("took {:.2f} seconds".format(time.time() - start))

    print("Building parser...", )
    start = time.time()
    parser = Parser(train_set,opt,dev_set,test_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Reading gold actions...")
    seq_train = read_seq(os.path.join(opt.datapath, opt.seqpath), parser,reduced,thr+3)

    print("Loading pretrained embeddings...", )
    start = time.time()

    if opt.withbert:
        embeddings_matrix = filter_bert(opt,parser)
    else:
        embeddings_matrix = filter_random(opt,parser)

    print("took {:.2f} seconds".format(time.time() - start))

    print("Vectorizing data...", )
    start = time.time()
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Preprocessing training data...", )
    start = time.time()
    train_examples = parser.create_instances(train_set,seq_train)
    
    print("took {:.2f} seconds".format(time.time() - start))
    return parser, embeddings_matrix, train_examples, train_set, dev_set, test_set, {'P':4}
    
# preprocess the test/evaluation data
def load_and_preprocess_data_test(opt,parser,reduced=True):
    
    print("Loading data...", )
    start = time.time()
    test_set = read_conll(
        os.path.join(opt.datapath, opt.testfile),
        lowercase=opt.lowercase)
    
    thr = 32
    if reduced:
        test_set = test_set[:thr+1]
    
    print("took {:.2f} seconds".format(time.time() - start))

    print("Building parser...", )
    start = time.time()
    print("took {:.2f} seconds".format(time.time() - start))

    if opt.withbert:
        embeddings_matrix = filter_bert(opt,parser)
    else:
        embeddings_matrix = filter_random(opt,parser)
        
    print("Vectorizing data...", )
    start = time.time()
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Preprocessing training data...", )
    start = time.time()
    print("took {:.2f} seconds".format(time.time() - start))
    return test_set, {'P':4}

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    pass
