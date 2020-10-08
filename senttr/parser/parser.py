# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers.configuration_bert import BertConfig
from parser.utils.base import BertBaseModel
from parser.utils.graph import BertGraphModel
from parser.utils.scalar_mix import ScalarMixWithDropout


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, n_labels):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.activation = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, n_labels)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, input_label):
        output = self.layer1(input_label)
        output = self.activation(output)
        output = self.layer2(output)

        return output

class Parser(nn.Module):

    def __init__(self, config, bertmodel):
        super(Parser, self).__init__()

        self.config = config

        # build and load BERT G2G model
        bertconfig = BertConfig.from_pretrained(
                config.main_path+"/model"+"/model_"+config.modelname+'/config.json')

        bertconfig.num_hidden_layers = config.n_attention_layer
        bertconfig.label_size = config.n_rels
        bertconfig.layernorm_value = config.layernorm_value
        bertconfig.layernorm_key = config.layernorm_key

        if self.config.input_graph:
            self.bert = BertGraphModel(bertconfig)
        else:
            self.bert = BertBaseModel(bertconfig)
        
        self.bert.load_state_dict(bertmodel.state_dict(),strict=False)
        self.mlp = Classifier(3*bertconfig.hidden_size,bertconfig.hidden_size,config.n_trans)
        self.mlp_rel = Classifier(2*bertconfig.hidden_size,bertconfig.hidden_size,config.n_rels)

        self.pad_index = config.pad_index
        self.unk_index = config.unk_index

    # build proper features for graph output mechanism
    def merge(self,states):
        features = [state.feature() for state in states]
        features = torch.stack(features)
        return features

    def merge_label(self,states):
        features = [state.feature_label() for state in states]
        features = torch.stack(features)
        return features

    # build graph input matrices for a batch
    def mix_graph(self,states):
        graphs = torch.stack([state.graph for state in states])
        labels = torch.stack([state.label for state in states])
        return graphs,labels

    def forward(self, words, tags, masks, states, actions=None, rels=None):

        mask = words.ne(self.pad_index)

        batch_size = words.size()[0]
        if actions is None:
            output_acts = torch.zeros((batch_size,self.config.max_seq,self.config.n_trans)).to(words.device)
            output_rels = torch.zeros((batch_size, self.config.max_seq,self.config.n_rels)).to(words.device)
            max_seq = self.config.max_seq
        else:
            output_acts = torch.zeros((batch_size,actions.size()[1],self.config.n_trans)).to(words.device)
            output_rels = torch.zeros((batch_size, actions.size()[1],self.config.n_rels)).to(words.device)
            max_seq = actions.size()[1]
        step = 0
        while step < max_seq:
            if self.config.input_graph:
                graphs,labels = self.mix_graph(states)
                embs = self.bert(words,tags,mask,graphs,labels)[0]
            else:
                embs = self.bert(words, tags, mask)[0]
            feats = self.merge(states)
            out = torch.stack([embs[i][feats[i]] for i in range(feats.size()[0])]).view(feats.size()[0],-1).clone()
            out_arc = self.mlp(out)

            feats_label = self.merge_label(states)
            out_label_input = torch.stack([embs[i][feats_label[i]] for i in range(feats_label.size()[0])])\
                .view(feats_label.size()[0],-1).clone()
            out_rel = self.mlp_rel(out_label_input)

            # mask PAD and BERT relations
            out_rel[:,0:2] = out_rel[:,0:2] - 1000
            output_acts[:,step] = out_arc
            output_rels[:,step] = out_rel

            # predict action and label for the next iteration (use gold during training)
            if actions is None:
                legal_actions = torch.tensor([state.legal_act() for state in states])\
                    .long().to(words.device)
                _,act = torch.max(out_arc+1000*legal_actions,dim=1)
                _,rel = torch.max(out_rel,dim=1)
            else:
                act = actions[:,step]
                rel = rels[:,step]

            for i,(state,a,r) in enumerate(zip(states,act,rel)):
                state.update(a,r)

            if all([state.finished() for state in states]) and actions is None:
                break
            step+=1

        if actions is None:
            return states
        else:
            return output_acts,output_rels
    @classmethod
    def load(cls, fname):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'],state['bertmodel'])

        parser.load_state_dict(state['state_dict'],strict=False)
        parser.to(device)

        return parser

    def save(self, fname):
        state = {
            'bertmodel':self.bert,
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)
