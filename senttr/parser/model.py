# -*- coding: utf-8 -*-

from parser.metric import Metric

import torch
import torch.nn as nn
import numpy as np
import numpy
from tqdm import tqdm
import torch.nn.functional as F


## this class keeps parser state information
class State(object):
    def __init__(self, mask,device,bert_label=None,input_graph=False):
        self.tok_buffer = mask.nonzero().squeeze(1)
        self.tok_stack = torch.zeros(len(self.tok_buffer)+1).long().to(device)
        self.tok_stack[0] = 1
        self.buf = [i+1 for i in range(len(self.tok_buffer))]
        self.stack = [0]
        self.head = [[-1, -1] for _ in range(len(self.tok_buffer)+1)]
        self.dict = {0:"LEFTARC", 1:"RIGHTARC" ,2:"SHIFT", 3:"SWAP"}
        self.graph,self.label,self.convert = self.build_graph(mask,device,bert_label)
        self.input_graph=input_graph

    # build partially constructed graph
    def build_graph(self,mask,device,bert_label):
        graph = torch.zeros((len(mask),len(mask))).long().to(device)
        label = torch.ones(len(mask) * bert_label).long().to(device)
        offset = self.tok_buffer.clone()
        convert = {0:1}
        convert.update({i+1:off.item() for i,off in enumerate(offset)})
        convert.update({len(convert):len(mask)})
        for i in range(len(offset)-1):
            graph[offset[i],offset[i]+1:offset[i+1]] = 1
            graph[offset[i]+1:offset[i+1],offset[i]] = 2
        label[offset] = 0
        label[:2] = 0
        del offset
        return graph,label,convert

    def get_graph(self):
        return self.graph,self.label

    #required features for graph output mechanism (exist classifier)
    def feature(self):
        return torch.cat((self.tok_stack[1].unsqueeze(0),self.tok_stack[0].unsqueeze(0)
                          ,self.tok_buffer[0].unsqueeze(0)))

    # required features for graph output mechanism (relation classifier)
    def feature_label(self):
        return torch.cat((self.tok_stack[1].unsqueeze(0),self.tok_stack[0].unsqueeze(0)))

    # update state
    def update(self,act,rel=None):
        act = self.dict[act.item()]
        if not self.finished():
            if act == "SHIFT":
                self.stack = [self.buf[0]] + self.stack
                self.buf = self.buf[1:]
                self.tok_buffer = torch.roll(self.tok_buffer,-1,dims=0).clone()
                self.tok_stack = torch.roll(self.tok_stack,1,dims=0).clone()
                self.tok_stack[0] = self.tok_buffer[-1].clone()
            elif act == "LEFTARC":
                self.head[self.stack[1]] = [self.stack[0], rel.item()]
                if self.input_graph:
                    self.graph[self.convert[self.stack[0]],self.convert[self.stack[1]]] = 1
                    self.graph[self.convert[self.stack[1]],self.convert[self.stack[0]]] = 2
                    self.label[self.convert[self.stack[1]]] = rel
                self.stack = [self.stack[0]] + self.stack[2:]
                self.tok_stack = torch.cat(
                    (self.tok_stack[0].unsqueeze(0),torch.roll(self.tok_stack[1:],-1,dims=0))).clone()
            elif act == "RIGHTARC":
                self.head[self.stack[0]] = [self.stack[1], rel.item()]
                if self.input_graph:
                    self.graph[self.convert[self.stack[1]],self.convert[self.stack[0]]] = 1
                    self.graph[self.convert[self.stack[0]],self.convert[self.stack[1]]] = 2
                    self.label[self.convert[self.stack[0]]] = rel
                self.stack = self.stack[1:]
                self.tok_stack = torch.roll(self.tok_stack,-1,dims=0).clone()
            elif act == "SWAP":
                self.buf = [self.stack[1]] + self.buf
                self.stack = [self.stack[0]] + self.stack[2:]
                self.tok_stack = torch.cat(
                    (self.tok_stack[0].unsqueeze(0), torch.roll(self.tok_stack[1:], -1, dims=0))).clone()
                self.tok_buffer = torch.roll(self.tok_buffer, 1, dims=0).clone()
                self.tok_buffer[0] = self.tok_stack[-1]

    # legal actions at evaluation time
    def legal_act(self):
        t = [0,0,0,0]
        if len(self.stack) >= 2 and self.stack[1] != 0:
            t[0] = 1
        if len(self.stack) >= 2 and self.stack[0] != 0:
            t[1] = 1
        if len(self.buf) > 0:
            t[2] = 1
        if len(self.stack) >= 2 and 0 < self.stack[1] < self.stack[0]:
            t[3] = 1
        return t

    # check whether the dependency tree is completed or not.
    def finished(self):
        return len(self.stack) == 1 and len(self.buf) == 0

    def __repr__(self):
        return "State:\nConvert:{}\n Graph:{}\n,Label:{}\nHead:{}\n".\
            format(self.convert,self.graph,self.label,self.head)

class Model(object):

    def __init__(self, vocab, parser, config, num_labels):
        super(Model, self).__init__()
        self.vocab = vocab
        self.parser = parser
        self.num_labels = num_labels
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

    def train(self, loader):
        self.parser.train()
        pbar = tqdm(total= len(loader))

        for ccc,(words, tags, masks, actions, mask_actions, rels) in enumerate(loader):
            
            states = [State(mask,tags.device,self.vocab.bert_index,self.config.input_graph)
                      for mask in masks]
            s_arc,s_rel = self.parser(words, tags, masks, states, actions, rels)


            if self.config.use_two_opts:
                self.optimizer_nonbert.zero_grad()
                self.optimizer_bert.zero_grad()
            else:
                self.optimizer.zero_grad()

            ## leftarc and rightarc have dependencies, so we filter swap and shift
            mask_rels = ((actions != 3).long() * (actions != 2).long() * mask_actions.long()).bool()

            actions = actions[mask_actions]
            s_arc = s_arc[mask_actions]

            rels = rels[mask_rels]
            s_rel = s_rel[mask_rels]

            loss = self.get_loss(s_arc,actions,s_rel,rels)
            loss.backward()
            ## optimization step
            if self.config.use_two_opts:
                self.optimizer_nonbert.step()
                self.optimizer_bert.step()
                self.scheduler_nonbert.step()
                self.scheduler_bert.step()
            else:
                self.optimizer.step()
                self.scheduler.step()
            del states,words,tags,masks,mask_actions,actions,rels,s_rel,s_arc,mask_rels

            pbar.update(1)

    @torch.no_grad()
    def evaluate(self, loader, punct=False):
        self.parser.eval()
        metric = Metric()
        pbar = tqdm(total=len(loader))

        for words, tags, masks,heads,rels,mask_heads in loader:
            states = [State(mask, tags.device,self.vocab.bert_index,self.config.input_graph)
                      for mask in masks]

            states = self.parser(words, tags, masks,states)

            pred_heads = []
            pred_rels = []
            for state in states:
                pred_heads.append([h[0] for h in state.head][1:])
                pred_rels.append([h[1] for h in state.head][1:])
            pred_heads = [item for sublist in pred_heads for item in sublist]
            pred_rels = [item for sublist in pred_rels for item in sublist]

            pred_heads = torch.tensor(pred_heads).to(heads.device)
            pred_rels = torch.tensor(pred_rels).to(heads.device)

            heads = heads[mask_heads]
            rels = rels[mask_heads]
            pbar.update(1)
            metric(pred_heads, pred_rels, heads, rels)
            del states

        return metric

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()
        metric = Metric()
        pbar = tqdm(total=len(loader))
        all_arcs, all_rels = [], []
        for words, tags, masks,heads,rels,mask_heads in loader:
            states = [State(mask, tags.device, self.vocab.bert_index,self.config.input_graph)
                      for mask in masks]
            states = self.parser(words, tags, masks, states)

            pred_heads = []
            pred_rels = []
            for state in states:
                pred_heads.append([h[0] for h in state.head][1:])
                pred_rels.append([h[1] for h in state.head][1:])

            pred_heads = [item for sublist in pred_heads for item in sublist]
            pred_rels = [item for sublist in pred_rels for item in sublist]

            pred_heads = torch.tensor(pred_heads).to(heads.device)
            pred_rels = torch.tensor(pred_rels).to(heads.device)

            heads = heads[mask_heads]
            rels = rels[mask_heads]

            metric(pred_heads, pred_rels, heads, rels)

            lens = masks.sum(1).tolist()

            all_arcs.extend(torch.split(pred_heads, lens))
            all_rels.extend(torch.split(pred_rels, lens))
            pbar.update(1)
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels, metric

    def get_loss(self, s_arc, actions, s_rel, rels):
        arc_loss = self.criterion(s_arc, actions)
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss