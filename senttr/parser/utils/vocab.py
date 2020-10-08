# -*- coding: utf-8 -*-

from collections import Counter
import os
import regex
import torch
from transformers import *

class Vocab(object):
    PAD = '[PAD]'
    UNK = '[UNK]'
    BERT = '[BERT]'

    def __init__(self, config, words, tags, rels):

        self.config = config
        self.max_pad_length = config.max_seq_length
        
        self.words = [self.PAD, self.UNK] + sorted(words)
        
        self.tags = sorted(tags)
        self.tags = [self.PAD, self.UNK] + ['<t>:'+tag for tag in self.tags]
        
        self.rels = sorted(rels)

        self.bert_index = 1
        if self.config.input_graph:
            self.rels = [self.PAD] + [self.BERT] + self.rels
        else:
            self.rels = [self.PAD] + self.rels

        ## left-arc:L,right-arc:R,shift:S,swap:H
        self.trans = ['L', 'R', 'S','H']
        self.trans_dict = {tr:i for i,tr in enumerate(self.trans)}

        self.word_dict = {word: i for i,word in enumerate(self.words)}
        self.punct = [word for word, i in self.word_dict.items() if regex.match(r'\p{P}+$', word)]
        ### Let's load a model and tokenizer############################

        self.bertmodel = BertModel.from_pretrained(config.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)

        self.tokenizer.add_tokens(self.tags + ['<ROOT>']+ self.punct + ['[CLS]']+ ['[SEP]'])

        all_tokens = self.tags + ['<ROOT>'] + self.punct + ['[CLS]'] + ['[SEP]'] + self.words
        all_pics = []
        for word in all_tokens:
            tokens = self.tokenizer.tokenize(word)
            for token in tokens:
                all_pics.append(token)
        self.word2bert = {}
        cou = 0
        for pic in all_pics:
            index = self.tokenizer.convert_tokens_to_ids(pic)
            if index not in self.word2bert:
                self.word2bert[index] = cou
                cou+= 1

        self.bertmodel.resize_token_embeddings(len(self.tokenizer))
        self.bertmodel.train()
        vectors = self.bertmodel.embeddings.word_embeddings.weight
        new_vectors = torch.index_select(vectors,0,torch.tensor(list(self.word2bert.keys())))
        self.bertmodel.resize_token_embeddings(len(self.word2bert))
        self.bertmodel.embeddings.word_embeddings = self.bertmodel.\
            embeddings.word_embeddings.from_pretrained(new_vectors)
        for index in self.word2bert:
            assert torch.all(torch.eq(self.bertmodel.embeddings.word_embeddings(
                torch.tensor(self.word2bert[index])),vectors[index])),\
                "index-word2bert:{}".format(vectors[index]-self.bertmodel.embeddings.
                                            word_embeddings(torch.tensor(self.word2bert[index])))

        # Train our model
        self.bertmodel.train()

        if os.path.exists(config.main_path + "/model" + "/model_" + config.modelname) != True:
            os.mkdir(config.main_path + "/model" + "/model_" + config.modelname)
            
        ### Now let's save our model and tokenizer to a directory
        self.bertmodel.save_pretrained(config.main_path + "/model" + "/model_" + config.modelname)

        self.tag_dict = {tag: i for i,tag in enumerate(self.tags)}
        
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}
        # ids of punctuation that appear in words

        self.puncts = []
        for punct in self.punct:
            self.puncts.append(self.word2bert.get(self.tokenizer.convert_tokens_to_ids(punct)))
        
        self.pad_index = self.tokenizer.convert_tokens_to_ids(self.PAD)
        self.pad_index = self.word2bert[self.pad_index]
        self.unk_index = self.tokenizer.convert_tokens_to_ids(self.UNK)
        self.unk_index = self.word2bert[self.unk_index]

        self.cls_index = self.word2bert[self.tokenizer.convert_tokens_to_ids('[CLS]') ]
        self.sep_index = self.word2bert[self.tokenizer.convert_tokens_to_ids('[SEP]')]

        self.n_words = len(self.words)
        self.n_tags = len(self.tags)
        self.n_rels = len(self.rels)
        self.n_trans = len(self.trans)
        self.n_train_words = self.n_words
        self.unk_count = 0
        self.total_count = 0
        self.long_seq = 0

    def __repr__(self):
        info = f"{self.__class__.__name__}: "
        info += f"{self.n_words} words, "
        info += f"{self.n_tags} tags, "
        info += f"{self.n_rels} rels"

        return info

    ## prepare data for train set
    def map_arcs_bert_pred(self, corpus, seq_corpus):

        all_words = []
        all_tags = []
        all_masks = []
        all_actions = []
        all_masks_action = []
        all_rels = []

        for i, (words, tags, seq) in enumerate(zip(corpus.words,corpus.tags, seq_corpus)):

            old_to_new_node = {0: 0}
            tokens_org, tokens_length = self.word2id(words)
            tokens = [item for sublist in tokens_org for item in sublist]

            index = 0
            for token_id, token_length in enumerate(tokens_length):
                index += token_length
                old_to_new_node[token_id + 1] = index

            # CLS heads and tags
            new_tags = []
            offsets = torch.tensor(list(old_to_new_node.values()))[:-1] + 1

            for token_id, token_length in enumerate(tokens_length):
                for sub_token in range(token_length):
                    new_tags.append(tags[token_id])

            words_id = torch.tensor([self.cls_index] + tokens + [self.sep_index])

            # 100 is the id of [UNK]
            self.unk_count += len((words_id == 100).nonzero())
            self.total_count += len(words_id)

            tags = torch.tensor([self.cls_index] + self.tag2id(new_tags) + [self.sep_index])

            masks = torch.zeros(len(words_id)).long()
            masks[offsets[1:]] = 1

            ## ignore some long sentences to fit the training phase in memory
            if len(seq['act']) < self.config.act_thr:
                all_words.append(words_id)
                all_tags.append(tags)
                all_masks.append(masks.bool())
                all_actions.append(torch.tensor(seq['act']))
                all_masks_action.append(torch.ones_like(torch.tensor(seq['act'])).bool())
                all_rels.append(torch.tensor(seq['rel']))

        print("Percentage of unkown tokens:{}".format(self.unk_count * 1.0 / self.total_count * 100))
        self.unk_count = 0
        self.total_count = 0

        return all_words, all_tags, all_masks, all_actions, all_masks_action, all_rels

    ## prepare data for test and evaluation
    def map_arcs_bert(self, corpus):
        all_words = []
        all_tags = []
        all_masks = []
        all_heads = []
        all_rels = []
        all_masks_head = []

        for i, (words, tags,heads,rels) in enumerate(zip(corpus.words, corpus.tags, corpus.heads, corpus.rels)):

            old_to_new_node = {0: 0}
            tokens_org, tokens_length = self.word2id(words)
            tokens = [item for sublist in tokens_org for item in sublist]

            index = 0
            for token_id, token_length in enumerate(tokens_length):
                index += token_length
                old_to_new_node[token_id + 1] = index

            # CLS heads and tags
            new_tags = []
            offsets = torch.tensor(list(old_to_new_node.values()))[:-1] + 1

            for token_id, token_length in enumerate(tokens_length):
                for sub_token in range(token_length):
                    new_tags.append(tags[token_id])

            words_id = torch.tensor([self.cls_index] + tokens + [self.sep_index])

            self.unk_count += len((words_id == self.unk_index).nonzero())
            self.total_count += len(words_id)

            tags = torch.tensor([self.cls_index] + self.tag2id(new_tags) + [self.sep_index])

            rels = torch.tensor([self.rel2id(rel) for rel in rels])
            masks = torch.zeros(len(words_id)).long()
            masks[offsets[1:]] = 1

            heads = torch.tensor(heads[1:])
            masks_head = torch.ones_like(heads)


            if len(masks) < 512:
                all_words.append(words_id)
                all_tags.append(tags)
                all_masks.append(masks.bool())
                all_rels.append(rels[1:])
                all_heads.append(heads)
                all_masks_head.append(masks_head.bool())

        print("Percentage of unknown tokens:{}".format(self.unk_count * 1.0 / self.total_count * 100))
        self.unk_count = 0
        self.total_count = 0

        return all_words, all_tags, all_masks, all_heads, all_rels,all_masks_head

    def word2id(self, sequence):
        WORD2ID = []
        lengths = []
        for word in sequence:
            x = self.tokenizer.tokenize(word)
            if len(x) == 0:
                x = ['[UNK]']
            x = self.tokenizer.convert_tokens_to_ids(x)
            x = [self.word2bert.get(y,self.unk_index) for y in x]
            lengths.append(len(x))
            WORD2ID.append(x)
        return WORD2ID,lengths   
    
    def tag2id(self, sequence):
        
        tags = []
        for tag in sequence:
            tokenized_tag = self.tokenizer.tokenize('<t>:'+tag)
            if len(tokenized_tag) != 1:
                tags.append(self.unk_index)
            else:
                tags.append(self.word2bert.get(self.tokenizer.convert_tokens_to_ids(
                    tokenized_tag)[0],self.unk_index))
        return tags

    def rel2id(self, rel):
        return self.rel_dict.get(rel, 0)

    def id2rel(self, ids):
        return [self.rels[i] for i in ids]

    def extend(self, words):
        self.words.extend(sorted(set(words).difference(self.word_dict)))
        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))
        self.n_words = len(self.words)


    def numericalize(self, corpus, seq_corpus = None):

        if seq_corpus is None:
            return self.map_arcs_bert(corpus)
        else:
            return self.map_arcs_bert_pred(corpus,seq_corpus)

    @classmethod
    def from_corpus(cls, config, corpus, corpus_dev=None, corpus_test=None, min_freq=0):
        if corpus_dev is not None:
            all_words = corpus.words + corpus_dev.words + corpus_test.words
        else:
            all_words = corpus.words
        words = Counter(word for seq in all_words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        tags = list({tag for seq in corpus.tags for tag in seq})
        rels = list({rel for seq in corpus.rels for rel in seq})
        vocab = cls(config, words, tags, rels)

        return vocab
