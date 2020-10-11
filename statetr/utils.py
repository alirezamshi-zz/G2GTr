#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging
import abc
import sys
import time
import numpy as np
import statistics

logger = logging.getLogger(__name__)


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

## prepare data for inputting to transformer
def prepare_data(ones,fgr,parser,batch_stack,batch_buffer,batch_stack_pos,batch_buffer_pos,
                batch_dep,batch_temp,batch_dep_pos,batch_pos_temp,batch_dep_label,
                batch_label_temp, mask_cls,mask_stack,mask_sep,mask_buffer,dep_buffer,dep_pos_buffer,
                dep_label_buffer,mask_delete=None,batch_del_label=None,
                batch_delete=None,batch_delete_pos=None):
    
    ### preparing the data for the input of Bert
    if fgr:
        
        batch_dep_input = torch.cat((ones*parser.CLS,batch_dep,ones*parser.SEP,dep_buffer,ones*parser.SEP,batch_temp),dim=1)
    
        batch_dep_pos_input = torch.cat((ones*parser.CLS,batch_dep_pos,ones*parser.SEP,dep_pos_buffer,
                                        ones*parser.SEP,batch_pos_temp),dim=1)

        batch_dep_label_input = torch.cat((ones*parser.L_NULL,batch_dep_label,ones*parser.L_NULL,dep_label_buffer,
                                          ones*parser.L_NULL,batch_label_temp),dim=1)
        
        batch_graph_label_input = torch.cat((ones*parser.L_NULL,batch_label_temp,ones*parser.L_NULL,ones*parser.L_NULL,
                                                 batch_label_temp,ones*parser.L_NULL,batch_del_label),dim=1)
        
        
        input_ids_x = torch.cat((ones*parser.CLS,batch_stack,ones*parser.SEP,
                             batch_buffer,ones*parser.SEP,batch_delete),dim=1)
        pos_ids_x = torch.cat((ones*parser.CLS,batch_stack_pos,ones*parser.SEP,
                           batch_buffer_pos,ones*parser.SEP,batch_delete_pos),dim=1)    
        
        attention_mask = torch.cat((mask_cls,mask_stack,mask_sep,mask_buffer,mask_sep,mask_delete),dim=1)
        
        return input_ids_x, pos_ids_x,batch_dep_input, batch_dep_pos_input,batch_dep_label_input,batch_graph_label_input \
               ,attention_mask
                                    
    else:

        batch_dep_input = torch.cat((ones*parser.CLS,batch_dep,ones*parser.SEP,dep_buffer,ones*parser.SEP),dim=1)
    
        batch_dep_pos_input = torch.cat((ones*parser.CLS,batch_dep_pos,ones*parser.SEP,dep_pos_buffer,ones*parser.SEP),dim=1)
    
        batch_dep_label_input = torch.cat((ones*parser.L_NULL,batch_dep_label,ones*parser.L_NULL,dep_label_buffer,
                                           ones*parser.L_NULL),dim=1)
        input_ids_x = torch.cat((ones*parser.CLS,batch_stack,ones*parser.SEP,
                             batch_buffer,ones*parser.SEP),dim=1)
        pos_ids_x = torch.cat((ones*parser.CLS,batch_stack_pos,ones*parser.SEP,
                           batch_buffer_pos,ones*parser.SEP),dim=1)    
        
        attention_mask = torch.cat((mask_cls,mask_stack,mask_sep,mask_buffer,mask_sep),dim=1)
        
        return input_ids_x, pos_ids_x, batch_dep_input, batch_dep_pos_input,batch_dep_label_input,attention_mask
   
    
    
def batched_index_select(t,dim,inds):
    
    dummy = inds.unsqueeze(2).expand(inds.size(0),inds.size(1),t.size(2))
    out = t.gather(dim,dummy)
    
    del dummy
        
    return out

## prepare the graph input matrix
def prepare_graph(graph_emb,stack_inds, buffer_inds,batch_size,graph_input,device,NULL):

    buffer_inds = buffer_inds.clone()
    buffer_inds[buffer_inds==NULL] = stack_inds.size()[1]
    total_inds = torch.cat(( stack_inds,stack_inds[:,0].unsqueeze(1),buffer_inds ),dim=1)
    assert not (len(total_inds[total_inds>stack_inds.size()[1] ]) > 0)

    filter_graph = batched_index_select(graph_emb,1,total_inds)
    
    graph_input[:,1:filter_graph.size()[1]+1,2*filter_graph.size()[2]+4:] = filter_graph
    
    filter_graph_t = filter_graph.transpose(1,2)
    
    graph_input[:,2*filter_graph.size()[2]+4:,1:filter_graph.size()[1]+1] = 2 * filter_graph_t
    
    mask_delete = torch.sum(graph_emb,dim=1).byte().to(device)

    return mask_delete,graph_input

# doing the swap operation
def tr_swap(stack_ind, stack, buffer_ind, buffer, stack_pos, buffer_pos, dep, dep_pos, mask_buffer,
             mask_stack, dep_label, pad_dep, pad_pos, pad_label,dep_buffer,dep_pos_buffer,dep_label_buffer):

    dependency_word_ind = stack_ind[:, -2].clone()
    dependency_word = stack[:, -2].clone()
    dependency_pos = stack_pos[:, -2].clone()
    dep_dependency = dep[:,-2].clone()
    dep_pos_dependency = dep_pos[:,-2].clone()
    dep_label_dependency = dep_label[:,-2].clone()

    ## change stack
    stack = torch.cat((torch.roll(stack[:, :-1], 1, dims=1), stack[:, -1].unsqueeze(0).transpose(1, 0)), dim=1)
    stack_ind[:, -2] = stack_ind.size()[1]
    stack_ind = torch.cat((torch.roll(stack_ind[:, :-1], 1, dims=1), stack_ind[:, -1].unsqueeze(0).transpose(1, 0)),
                          dim=1)
    stack_pos = torch.cat((torch.roll(stack_pos[:, :-1], 1, dims=1), stack_pos[:, -1].unsqueeze(0).transpose(1, 0)),
                          dim=1)
    mask_stack = torch.cat((torch.roll(mask_stack[:, :-1], 1, dims=1), mask_stack[:, -1].unsqueeze(0).transpose(1, 0)),
                           dim=1)
    mask_stack[:, 0] = 0

    dep = torch.cat((torch.roll(dep[:, :-1], 1, dims=1), dep[:, -1].unsqueeze(0).transpose(1, 0)), dim=1)
    dep_pos = torch.cat((torch.roll(dep_pos[:, :-1], 1, dims=1), dep_pos[:, -1].unsqueeze(0).transpose(1, 0)), dim=1)
    dep_label = torch.cat((torch.roll(dep_label[:, :-1], 1, dims=1), dep_label[:, -1].unsqueeze(0).transpose(1, 0)),
                          dim=1)

    ##change buffer
    mask_buffer = torch.roll(mask_buffer,1,dims=1)
    buffer = torch.roll(buffer,1,dims=1)

    buffer_ind = torch.roll(buffer_ind,1,dims=1)
    buffer_pos = torch.roll(buffer_pos,1,dims=1)

    mask_buffer[:,0] = 1
    buffer_ind[:,0] = dependency_word_ind
    buffer[:,0] = dependency_word
    buffer_pos[:,0] = dependency_pos

    dep_buffer = torch.roll(dep_buffer,1,dims=1)
    dep_pos_buffer = torch.roll(dep_pos_buffer,1,dims=1)
    dep_label_buffer = torch.roll(dep_label_buffer,1,dims=1)

    dep_buffer[:,0] = dep_dependency
    dep_pos_buffer[:,0] = dep_pos_dependency
    dep_label_buffer[:,0] = dep_label_dependency



    return stack_ind, stack,buffer_ind, buffer, stack_pos, buffer_pos, dep, dep_pos, mask_buffer, mask_stack,\
           dep_label, dep_buffer,dep_pos_buffer,dep_label_buffer

# doing the shift operation
def tr_shift(stack_ind, stack, buffer_ind, buffer, stack_pos, buffer_pos, dep,dep_pos,mask_buffer,
                 mask_stack, dep_label, pad_dep, pad_pos,pad_label,dep_buffer,dep_pos_buffer,dep_label_buffer):
    
    ## save data
    word = buffer[:,0].clone()
    word_ind = buffer_ind[:,0].clone()
    pos = buffer_pos[:,0].clone()
    dep_first_word_buffer = dep_buffer[:,0].clone()
    dep_first_pos_buffer = dep_pos_buffer[:,0].clone()
    dep_first_label_buffer = dep_label_buffer[:,0].clone()
            
    ## buffer change
    mask_buffer[:,0] = 0
    mask_buffer = torch.roll(mask_buffer,-1,dims=1)
    buffer = torch.roll(buffer,-1,dims=1)
    buffer_ind = torch.roll(buffer_ind,-1,dims=1)
    buffer_ind[:,-1] = buffer_ind.size()[1]+1
    buffer_pos = torch.roll(buffer_pos,-1,dims=1)

    dep_buffer = torch.roll(dep_buffer,1,dims=1)
    dep_pos_buffer = torch.roll(dep_pos_buffer,1,dims=1)
    dep_label_buffer = torch.roll(dep_label_buffer,1,dims=1)

    dep_buffer[:,-1] = pad_dep
    dep_pos_buffer[:,-1] = pad_pos
    dep_label_buffer[:,-1] = pad_label

            
    ## stack change
    stack = torch.roll(stack,-1,dims=1)
    stack[:,-1] = word

    stack_ind = torch.roll(stack_ind,-1,dims=1)
    stack_ind[:,-1] = word_ind
    
    mask_stack = torch.roll(mask_stack,-1,dims=1)
    mask_stack[:,-1] = 1
        
    stack_pos = torch.roll(stack_pos,-1)
    stack_pos[:,-1] = pos
            
    ### dep change
    dep = torch.roll(dep,-1)
    dep[:,-1] = dep_first_word_buffer
    dep_pos = torch.roll(dep_pos,-1)
    dep_pos[:,-1] = dep_first_pos_buffer
    dep_label = torch.roll(dep_label,-1)
    dep_label[:,-1] = dep_first_label_buffer
        
    del word,pos,word_ind    
    
    return stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos,dep,dep_pos,mask_buffer,mask_stack, dep_label,\
           dep_buffer, dep_pos_buffer, dep_label_buffer

# doing the left arc operation
def tr_left_arc(stack_ind, stack, buffer_ind, buffer, stack_pos, buffer_pos, dep,dep_pos,mask_buffer,
                mask_stack, dep_label, label, pad_dep, pad_pos, pad_label,dep_buffer, dep_pos_buffer,
                dep_label_buffer,graph_emb=None,del_label=None):

    ## save data
    dependency_word_ind = stack_ind[:,-2].clone()
    head_word_ind = stack_ind[:,-1].clone()
    
    dependency_word = stack[:,-2].clone()
    head_word = stack[:,-1].clone()
    
    dependency_pos = stack_pos[:,-2].clone()
    head_pos = stack_pos[:,-1].clone()
          
    if graph_emb is not None:
        graph_emb[torch.arange(dependency_word_ind.size()[0]),head_word_ind, dependency_word_ind - 1] = 1
        del_label.scatter_(1,(dependency_word_ind-1).unsqueeze(0).transpose(1,0),label.unsqueeze(0).transpose(1,0))
        
    ## change stack
    stack = torch.cat((torch.roll(stack[:,:-1],1,dims=1),stack[:,-1].unsqueeze(0).transpose(1,0)),dim=1)
    stack_ind[:,-2] = stack_ind.size()[1]
    stack_ind = torch.cat((torch.roll(stack_ind[:,:-1],1,dims=1),stack_ind[:,-1].unsqueeze(0).transpose(1,0)),dim=1)
    stack_pos = torch.cat((torch.roll(stack_pos[:,:-1],1,dims=1),stack_pos[:,-1].unsqueeze(0).transpose(1,0)),dim=1)
    mask_stack = torch.cat((torch.roll(mask_stack[:,:-1],1,dims=1),mask_stack[:,-1].unsqueeze(0).transpose(1,0)),dim=1)
    mask_stack[:,0] = 0
    
    dep = torch.cat((torch.roll(dep[:,:-1],1,dims=1),dep[:,-1].unsqueeze(0).transpose(1,0)),dim=1)
    dep[:,-1] = dependency_word
    
    dep_pos = torch.cat((torch.roll(dep_pos[:,:-1],1,dims=1),dep_pos[:,-1].unsqueeze(0).transpose(1,0)),dim=1)
    dep_pos[:,-1] = dependency_pos    
    
    dep_label = torch.cat((torch.roll(dep_label[:,:-1],1,dims=1),dep_label[:,-1].unsqueeze(0).transpose(1,0)),dim=1)
    dep_label[:,-1] = label   
    
    return stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos, dep,dep_pos,mask_buffer,\
           mask_stack,dep_label,label,graph_emb,del_label,head_word_ind,dependency_word_ind, dep_buffer, dep_pos_buffer,\
           dep_label_buffer

# doing right arc operation
def tr_right_arc(stack_ind, stack, buffer_ind, buffer, stack_pos, buffer_pos, dep,dep_pos,mask_buffer,
                 mask_stack, dep_label, label, pad_dep, pad_pos, pad_label,dep_buffer,dep_pos_buffer,dep_label_buffer,
                 graph_emb=None,del_label=None):

    dependency_word_ind = stack_ind[:,-1].clone()
    head_word_ind = stack_ind[:,-2].clone()
    
    dependency_word = stack[:,-1].clone()
    head_word = stack[:,-2].clone()
    
    dependency_pos = stack_pos[:,-1].clone()
    head_pos = stack_pos[:,-2].clone()
    
    if graph_emb is not None:
        graph_emb[torch.arange(dependency_word_ind.size()[0]),head_word_ind, dependency_word_ind - 1] = 1
        del_label.scatter_(1,(dependency_word_ind-1).unsqueeze(0).transpose(1,0),label.unsqueeze(0).transpose(1,0))
        
    ## change stack 
    stack = torch.cat((stack[:,-1].unsqueeze(0).transpose(1,0),stack[:,:-1]),dim=1)
    stack_ind[:,-1] = stack_ind.size()[1]
    stack_ind = torch.cat((stack_ind[:,-1].unsqueeze(0).transpose(1,0),stack_ind[:,:-1]),dim=1)
    stack_pos = torch.cat((stack_pos[:,-1].unsqueeze(0).transpose(1,0),stack_pos[:,:-1]),dim=1)
    mask_stack = torch.cat((mask_stack[:,-1].unsqueeze(0).transpose(1,0),mask_stack[:,:-1]),dim=1)
        
    mask_stack[:,0] = 0

    dep = torch.cat((dep[:,-1].unsqueeze(0).transpose(1,0),dep[:,:-1]),dim=1)
    dep[:,-1] = dependency_word
    
    dep_pos = torch.cat((dep_pos[:,-1].unsqueeze(0).transpose(1,0),dep_pos[:,:-1]),dim=1)
    dep_pos[:,-1] = dependency_pos    
    
    dep_label = torch.cat((dep_label[:,-1].unsqueeze(0).transpose(1,0),dep_label[:,:-1]),dim=1)
    dep_label[:,-1] = label     

        
    return stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos, dep,dep_pos,mask_buffer,\
           mask_stack,dep_label,label,graph_emb,del_label,head_word_ind,dependency_word_ind,dep_buffer, dep_pos_buffer,\
           dep_label_buffer
# split input data based on predicted action
def filter_update(mask,stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos,dep,dep_pos,mask_buffer,
                  mask_stack,dep_label, label,graph_emb,del_label,dep_buffer,dep_pos_buffer,dep_label_buffer):

    stack_ind = stack_ind[mask]
    stack = stack[mask]
    buffer_ind = buffer_ind[mask]
    buffer = buffer[mask]
    stack_pos = stack_pos[mask]
    buffer_pos = buffer_pos[mask]
    dep = dep[mask]
    dep_pos = dep_pos[mask]
    mask_buffer = mask_buffer[mask]
    mask_stack = mask_stack[mask]
    dep_label = dep_label[mask]
    label = label[mask]
    dep_buffer = dep_buffer[mask]
    dep_pos_buffer = dep_pos_buffer[mask]
    dep_label_buffer = dep_label_buffer[mask]

    if graph_emb is not None:
        graph_emb = graph_emb[mask]
        del_label = del_label[mask]


    return stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos,dep,dep_pos,mask_buffer,\
           mask_stack,dep_label, label,graph_emb,del_label,dep_buffer,dep_pos_buffer,dep_label_buffer

def convert_back(index, element1, element2, element3, element4, element5):

    element = torch.cat((element1,element2,element3,element4,element5))
    element = element.index_select(0,index)
    
    return element


def create_dependencies(index_l,index_r,head_word_ind_l,dependency_word_ind_l,label_l,head_word_ind_r,
                        dependency_word_ind_r,label_r,dependencies,pad_label):

    if len(index_l)==0 and len(index_r)==0:
        return dependencies
    index = torch.cat((index_l,index_r)).squeeze(1)
    head_word = torch.cat((head_word_ind_l,head_word_ind_r))
    dep_word = torch.cat((dependency_word_ind_l,dependency_word_ind_r))
    label = torch.cat((label_l,label_r))
    bias = int(pad_label/2)
    i = 0
    for item in index:
        if i < len(index_l):
            label_temp = label[i].item()
        else:
            label_temp = label[i].item() - bias
            
        dependencies[item].append( (head_word[i].item(), dep_word[i].item(), label_temp) ) 
        i += 1
        
    return dependencies


## main function for updating parser states
def update_state(mode,opt,stack_ind, stack, buffer_ind, buffer, stack_pos, buffer_pos, dep,dep_pos,mask_buffer,
                 mask_stack, transition, dep_label, label, pad_dep, pad_pos,
                 pad_label,dep_buffer, dep_pos_buffer, dep_label_buffer,
                 graph_emb=None,del_label=None,dependencies=None):

    mask_s = (transition == 2) #shift
    mask_l = (transition == 0) #left-arc
    mask_r = (transition == 1) #right-arc
    mask_h = (transition == 3) #swap
    mask_t = (transition == 4) # None

    index_s = mask_s.nonzero()
    index_l = mask_l.nonzero()
    index_r = mask_r.nonzero()
    index_h = mask_h.nonzero()
    index_t = mask_t.nonzero()

    total_index = torch.cat((index_s,index_l,index_r,index_h,index_t)).squeeze(1)
    right_index = total_index.argsort(dim=0)
    
    #### filtering process
    ## filter shift
    stack_ind_s,stack_s,buffer_ind_s,buffer_s,stack_pos_s,buffer_pos_s,dep_s,dep_pos_s,mask_buffer_s,\
    mask_stack_s,dep_label_s,label_s,graph_emb_s,del_label_s,dep_buffer_s,dep_pos_buffer_s,dep_label_buffer_s =\
    filter_update(mask_s,stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos,dep,dep_pos,mask_buffer,
                  mask_stack,dep_label, label,graph_emb,del_label,dep_buffer,dep_pos_buffer,dep_label_buffer)
    
    ## filter left-arc
    stack_ind_l,stack_l,buffer_ind_l,buffer_l,stack_pos_l,buffer_pos_l,dep_l,dep_pos_l,mask_buffer_l,\
    mask_stack_l,dep_label_l,label_l,graph_emb_l,del_label_l,dep_buffer_l,dep_pos_buffer_l,dep_label_buffer_l=\
    filter_update(mask_l,stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos,dep,dep_pos,mask_buffer,
                  mask_stack,dep_label, label,graph_emb,del_label,dep_buffer,dep_pos_buffer,dep_label_buffer)
    
    ## filter right-arc
    stack_ind_r,stack_r,buffer_ind_r,buffer_r,stack_pos_r,buffer_pos_r,dep_r,dep_pos_r,mask_buffer_r,\
    mask_stack_r,dep_label_r,label_r,graph_emb_r,del_label_r,dep_buffer_r,dep_pos_buffer_r,dep_label_buffer_r =\
    filter_update(mask_r,stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos,dep,dep_pos,mask_buffer,
                  mask_stack,dep_label, label,graph_emb,del_label,dep_buffer,dep_pos_buffer,dep_label_buffer)

    ## filter swap
    stack_ind_h,stack_h,buffer_ind_h,buffer_h,stack_pos_h,buffer_pos_h,dep_h,dep_pos_h,mask_buffer_h,\
    mask_stack_h,dep_label_h,label_h,graph_emb_h,del_label_h,dep_buffer_h,dep_pos_buffer_h,dep_label_buffer_h =\
    filter_update(mask_h,stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos,dep,dep_pos,mask_buffer,
                  mask_stack,dep_label, label,graph_emb,del_label,dep_buffer,dep_pos_buffer,dep_label_buffer)

    ## filter none
    stack_ind_t,stack_t,buffer_ind_t,buffer_t,stack_pos_t,buffer_pos_t,dep_t,dep_pos_t,mask_buffer_t,\
    mask_stack_t,dep_label_t,label_t,graph_emb_t,del_label_t,dep_buffer_t,dep_pos_buffer_t,dep_label_buffer_t=\
    filter_update(mask_t,stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos,dep,dep_pos,mask_buffer,
                  mask_stack,dep_label, label,graph_emb,del_label,dep_buffer,dep_pos_buffer,dep_label_buffer)
    
    
    ###### action part
    head_word_ind_l = torch.tensor([]).long().to(stack_ind.device)
    dependency_word_ind_l = torch.tensor([]).long().to(stack_ind.device)
    head_word_ind_r = torch.tensor([]).long().to(stack_ind.device)
    dependency_word_ind_r = torch.tensor([]).long().to(stack_ind.device)
    
    #### do actions
    if len(index_s):
        stack_ind_s,stack_s,buffer_ind_s,buffer_s,stack_pos_s,buffer_pos_s,dep_s,dep_pos_s,mask_buffer_s,mask_stack_s,\
        dep_label_s,dep_buffer_s,dep_pos_buffer_s,dep_label_buffer_s = tr_shift(stack_ind_s, stack_s, buffer_ind_s,
                    buffer_s, stack_pos_s, buffer_pos_s, dep_s,dep_pos_s,mask_buffer_s,mask_stack_s, dep_label_s,
                    pad_dep, pad_pos,pad_label,dep_buffer_s,dep_pos_buffer_s,dep_label_buffer_s)

    if len(index_h):
        stack_ind_h,stack_h,buffer_ind_h,buffer_h,stack_pos_h,buffer_pos_h,dep_h,dep_pos_h,mask_buffer_h,mask_stack_h,\
        dep_label_h,dep_buffer_h,dep_pos_buffer_h,dep_label_buffer_h = tr_swap(stack_ind_h, stack_h, buffer_ind_h,
                    buffer_h, stack_pos_h, buffer_pos_h, dep_h,dep_pos_h,mask_buffer_h,mask_stack_h, dep_label_h,
                    pad_dep, pad_pos,pad_label,dep_buffer_h,dep_pos_buffer_h,dep_label_buffer_h)
    
    if len(index_l):
        stack_ind_l,stack_l,buffer_ind_l,buffer_l,stack_pos_l,buffer_pos_l, dep_l,dep_pos_l,mask_buffer_l,\
        mask_stack_l,dep_label_l,label_l,graph_emb_l,del_label_l,head_word_ind_l,dependency_word_ind_l,dep_buffer_l,\
        dep_pos_buffer_l,dep_label_buffer_l = tr_left_arc(stack_ind_l, stack_l, buffer_ind_l, buffer_l, stack_pos_l,
                    buffer_pos_l, dep_l,dep_pos_l,mask_buffer_l,mask_stack_l, dep_label_l, label_l,
                    pad_dep, pad_pos, pad_label,dep_buffer_l,dep_pos_buffer_l, dep_label_buffer_l,graph_emb_l,del_label_l)
        
    if len(index_r):
        stack_ind_r,stack_r,buffer_ind_r,buffer_r,stack_pos_r,buffer_pos_r, dep_r,dep_pos_r,mask_buffer_r,\
        mask_stack_r,dep_label_r,label_r,graph_emb_r,del_label_r,head_word_ind_r,dependency_word_ind_r,dep_buffer_r,\
        dep_pos_buffer_r,dep_label_buffer_r = tr_right_arc(stack_ind_r, stack_r, buffer_ind_r, buffer_r,stack_pos_r,
                    buffer_pos_r, dep_r,dep_pos_r,mask_buffer_r,mask_stack_r, dep_label_r, label_r, pad_dep,
                    pad_pos, pad_label,dep_buffer_r,dep_pos_buffer_r,dep_label_buffer_r,graph_emb_r,del_label_r)
    
    if mode:
        dependencies = create_dependencies(index_l,index_r,head_word_ind_l,dependency_word_ind_l,label_l,head_word_ind_r,
                                           dependency_word_ind_r,label_r,dependencies,pad_label)

    stack_ind = convert_back(right_index,stack_ind_s,stack_ind_l,stack_ind_r,stack_ind_h,stack_ind_t)
    stack = convert_back(right_index,stack_s,stack_l,stack_r,stack_h,stack_t)
    stack_pos = convert_back(right_index,stack_pos_s,stack_pos_l,stack_pos_r,stack_pos_h,stack_pos_t)
    mask_stack = convert_back(right_index,mask_stack_s,mask_stack_l,mask_stack_r,mask_stack_h,mask_stack_t)
    
    buffer_ind = convert_back(right_index,buffer_ind_s,buffer_ind_l,buffer_ind_r,buffer_ind_h,buffer_ind_t)
    buffer = convert_back(right_index,buffer_s,buffer_l,buffer_r,buffer_h,buffer_t)
    buffer_pos = convert_back(right_index,buffer_pos_s,buffer_pos_l,buffer_pos_r,buffer_pos_h,buffer_pos_t)
    mask_buffer = convert_back(right_index,mask_buffer_s,mask_buffer_l,mask_buffer_r,mask_buffer_h,mask_buffer_t)

    dep = convert_back(right_index,dep_s,dep_l,dep_r,dep_h,dep_t)
    dep_pos = convert_back(right_index,dep_pos_s,dep_pos_l,dep_pos_r,dep_pos_h,dep_pos_t)
    dep_label = convert_back(right_index,dep_label_s,dep_label_l,dep_label_r,dep_label_h,dep_label_t)

    dep_buffer = convert_back(right_index,dep_buffer_s,dep_buffer_l,dep_buffer_r,dep_buffer_h,dep_buffer_t)
    dep_pos_buffer = convert_back(right_index,dep_pos_buffer_s,dep_pos_buffer_l,dep_pos_buffer_r,dep_pos_buffer_h,
                                  dep_pos_buffer_t)
    dep_label_buffer = convert_back(right_index,dep_label_buffer_s,dep_label_buffer_l,dep_label_buffer_r,
                                    dep_label_buffer_h,dep_label_buffer_t)
    
    if opt.graphinput:
        graph_emb = convert_back(right_index,graph_emb_s,graph_emb_l,graph_emb_r,graph_emb_h,graph_emb_t)
        del_label = convert_back(right_index,del_label_s,del_label_l,del_label_r,del_label_h,del_label_t)
    

    if mode:
        return stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos,dep,dep_pos,mask_buffer,mask_stack, dep_label,\
               graph_emb,del_label,dep_buffer,dep_pos_buffer,dep_label_buffer,dependencies
    else:
        return stack_ind,stack,buffer_ind,buffer,stack_pos,buffer_pos,dep,dep_pos,mask_buffer,mask_stack, dep_label,\
               graph_emb,del_label,dep_buffer,dep_pos_buffer,dep_label_buffer

## making train batch from input data
def batch_train(dataset, batch_size, pad, parser):
    
    #### initialization #############
    pad = pad['P']
    pad_word = parser.NULL
    pad_pos = parser.P_NULL
    pad_label = parser.L_NULL
    gold_actions = []
    train_data = []
    gold_labels = []
    max_seq_length = 0
    dataset.sort(key=lambda row: len(row[1]), reverse=True)
    ## split the gold actions and sentences
    for i, example in enumerate(dataset):
        train_data.append(example[0])
        gold_actions.append(example[1])
        gold_labels.append(example[2])

    instances = []
    list_seq_len = []
    for i in range(0, len(dataset), batch_size):
        
        batch_data = train_data[i : min(i + batch_size,len(dataset))]
        bs = len(batch_data)

        ### batch buffer 
        seq_len_data = max([len(batch_data[i]['word'])-1 for i in range(len(batch_data))])  # find longest seq
        
        batch_buffer_tensor = (torch.ones((bs,seq_len_data)) * pad_word).long()

        batch_buffer_pos_tensor = (torch.ones((bs,seq_len_data)) * pad_pos).long()
        
        mask_buffer_tensor = torch.zeros((bs,seq_len_data)).byte()
        
        batch_buffer_ind = (torch.ones((bs,seq_len_data)) * pad_word).long()
        
        for idx, bdata in enumerate(batch_data):
            
            batch_buffer_ind[idx,0:len(bdata['word'])-1] = torch.arange(1,len(bdata['word']))
            
            batch_buffer_tensor[idx,0:len(bdata['word'])-1] = torch.LongTensor(bdata['word'][1:])

            batch_buffer_pos_tensor[idx,0:len(bdata['pos'])-1] = torch.LongTensor(bdata['pos'][1:])
            
            mask_buffer_tensor[idx,0:len(bdata['word'])-1] = 1        
        
        batch_actions = gold_actions[i : min(i + batch_size,len(dataset))]
        
        batch_labels = gold_labels[i : min(i + batch_size,len(dataset))]
        

        ### batch the gold actions
        seq_len_ac = max([len(batch_actions[i]) for i in range(len(batch_actions))])  # find longest seq
        if seq_len_ac > max_seq_length:
            max_seq_length = seq_len_ac
            
        list_seq_len.append(seq_len_ac)
            
        batch_label_tensor = (torch.ones((bs,seq_len_ac)) * pad_label).long()
        
        mask_label_tensor = torch.zeros((bs,seq_len_ac)).byte()
        
        batch_actions_tensor = (torch.ones((bs,seq_len_ac)) * pad).long()
        
        mask_tensor = torch.zeros((bs,seq_len_ac)).byte()
        
        for idx, (batch_action, batch_label) in enumerate(zip(batch_actions,batch_labels)):
            
            batch_label_tensor[idx,0:len(batch_action)] = torch.LongTensor(batch_label)
            batch_actions_tensor[idx,0:len(batch_action)] = torch.LongTensor(batch_action)
            mask_tensor[idx,0:len(batch_action)] = 1
        
        mask_label_tensor = (batch_label_tensor != pad_label).byte()

        instances.append((batch_buffer_ind, batch_buffer_tensor, batch_buffer_pos_tensor,
                        mask_buffer_tensor,batch_actions_tensor,mask_tensor, batch_label_tensor,
                        mask_label_tensor))
        
    del batch_label_tensor,mask_label_tensor,batch_actions_tensor,mask_tensor,batch_buffer_tensor,batch_buffer_ind,\
        batch_buffer_pos_tensor, mask_buffer_tensor
    
    mean_seq_length = statistics.mean(list_seq_len)
    
    return instances,max_seq_length,mean_seq_length

## making dev/test batch from input data
def batch_dev_test(dataset, batch_size, pad_word, pad_pos, parser, no_sort=False):

    instances = []
    if not no_sort:
        dataset.sort(key=lambda row: len(row['word']), reverse=True)

    counter = 0

    for i in range(0, len(dataset), batch_size):
        
        batch_data = dataset[i : min(i + batch_size,len(dataset)) ]
        
        bs = len(batch_data)
        ### batch buffer 
        seq_len_data = max([len(batch_data[i]['word'])-1 for i in range(len(batch_data))])  # find longest seq
        
        batch_buffer_tensor = (torch.ones((bs,seq_len_data)) * pad_word).long()

        batch_buffer_pos_tensor = (torch.ones((bs,seq_len_data)) * pad_pos).long()
        
        mask_buffer_tensor = torch.zeros((bs,seq_len_data)).byte()
        
        batch_buffer_ind = (torch.ones((bs,seq_len_data)) * pad_word).long()
        
        for idx, bdata in enumerate(batch_data):
            
            batch_buffer_ind[idx,0:len(bdata['word'])-1] = torch.arange(1,len(bdata['word']))
            
            batch_buffer_tensor[idx,0:len(bdata['word'])-1] = torch.LongTensor(bdata['word'][1:])

            batch_buffer_pos_tensor[idx,0:len(bdata['pos'])-1] = torch.LongTensor(bdata['pos'][1:])
            
            mask_buffer_tensor[idx,0:len(bdata['word'])-1] = 1

        instances.append((batch_buffer_ind, batch_buffer_tensor,
                          batch_buffer_pos_tensor, mask_buffer_tensor))

    del batch_buffer_tensor,batch_buffer_pos_tensor,mask_buffer_tensor,batch_buffer_ind
    
    return instances


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t

def relative_matmul_dp(x,z):
    """ Helper function for dependency parsing relations"""
    x_t = x.transpose(1,2)
    z_t = z.transpose(2,3)
    
    out = torch.matmul(x_t,z_t)
    
    out = out.transpose(1,2)
    
    return out


def relative_matmul_dpv(x,z):
    """ Helper function for dependency parsing relations"""

    x = x.transpose(1,2)
    out = torch.matmul(x,z)
    out = out.transpose(1,2)
    
    return out
    
class _LRSchedule(ABC):
    """ Parent of all LRSchedules here. """
    warn_t_total = False        # is set to True for schedules where progressing beyond t_total steps doesn't make sense
    def __init__(self, warmup=0.002, t_total=-1, **kw):
        """
        :param warmup:  what fraction of t_total steps will be used for linear warmup
        :param t_total: how many training steps (updates) are planned
        :param kw:
        """
        super(_LRSchedule, self).__init__(**kw)
        if t_total < 0:
            logger.warning("t_total value of {} results in schedule not being applied".format(t_total))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        warmup = max(warmup, 0.)
        self.warmup, self.t_total = float(warmup), float(t_total)
        self.warned_for_t_total_at_progress = -1

    def get_lr(self, step, nowarn=False):
        """
        :param step:    which of t_total steps we're on
        :param nowarn:  set to True to suppress warning regarding training beyond specified 't_total' steps
        :return:        learning rate multiplier for current update
        """
        if self.t_total < 0:
            return 1.
        progress = float(step) / self.t_total
        ret = self.get_lr_(progress)
        # warning for exceeding t_total (only active with warmup_linear
        if not nowarn and self.warn_t_total and progress > 1. and progress > self.warned_for_t_total_at_progress:
            logger.warning(
                "Training beyond specified 't_total'. Learning rate multiplier set to {}. Please set 't_total' of {} correctly."
                    .format(ret, self.__class__.__name__))
            self.warned_for_t_total_at_progress = progress
        # end warning
        return ret

    @abc.abstractmethod
    def get_lr_(self, progress):
        """
        :param progress:    value between 0 and 1 (unless going beyond t_total steps) specifying training progress
        :return:            learning rate multiplier for current update
        """
        return 1.


class ConstantLR(_LRSchedule):
    def get_lr_(self, progress):
        return 1.


class WarmupCosineSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Decreases learning rate from 1. to 0. over remaining `1 - warmup` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    warn_t_total = True
    def __init__(self, warmup=0.002, t_total=-1, cycles=.5, **kw):
        """
        :param warmup:      see LRSchedule
        :param t_total:     see LRSchedule
        :param cycles:      number of cycles. Default: 0.5, corresponding to cosine decay from 1. at progress==warmup and 0 at progress==1.
        :param kw:
        """
        super(WarmupCosineSchedule, self).__init__(warmup=warmup, t_total=t_total, **kw)
        self.cycles = cycles

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)   # progress after warmup
            return 0.5 * (1. + math.cos(math.pi * self.cycles * 2 * progress))


class WarmupCosineWithHardRestartsSchedule(WarmupCosineSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
    learning rate (with hard restarts).
    """
    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kw):
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(warmup=warmup, t_total=t_total, cycles=cycles, **kw)
        assert(cycles >= 1.)

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)     # progress after warmup
            ret = 0.5 * (1. + math.cos(math.pi * ((self.cycles * progress) % 1)))
            return ret


class WarmupCosineWithWarmupRestartsSchedule(WarmupCosineWithHardRestartsSchedule):
    """
    All training progress is divided in `cycles` (default=1.) parts of equal length.
    Every part follows a schedule with the first `warmup` fraction of the training steps linearly increasing from 0. to 1.,
    followed by a learning rate decreasing from 1. to 0. following a cosine curve.
    """
    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kw):
        assert(warmup * cycles < 1.)
        warmup = warmup * cycles if warmup >= 0 else warmup
        super(WarmupCosineWithWarmupRestartsSchedule, self).__init__(warmup=warmup, t_total=t_total, cycles=cycles, **kw)

    def get_lr_(self, progress):
        progress = progress * self.cycles % 1.
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)     # progress after warmup
            ret = 0.5 * (1. + math.cos(math.pi * progress))
            return ret


class WarmupConstantSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Keeps learning rate equal to 1. after warmup.
    """
    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1.


class WarmupLinearSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `1 - warmup` steps.
    """
    warn_t_total = True
    def get_lr_(self, progress):
        #print(progress)
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)
        #f1 = open("./test.txt","a")
        #f1.write(str(x)+"\n")
        #f1.close()
        #return x

SCHEDULES = {
    None:       ConstantLR,
    "none":     ConstantLR,
    "warmup_cosine": WarmupCosineSchedule,
    "warmup_constant": WarmupConstantSchedule,
    "warmup_linear": WarmupLinearSchedule
}


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate of 1. (no warmup regardless of warmup setting). Default: -1
        schedule: schedule to use for the warmup (see above).
            Can be `'warmup_linear'`, `'warmup_constant'`, `'warmup_cosine'`, `'none'`, `None` or a `_LRSchedule` object (see below).
            If `None` or `'none'`, learning rate is always kept constant.
            Default : `'warmup_linear'`
        betas: Adams betas. Default: (0.9, 0.999)
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), e=1e-6, weight_decay=0.01, max_grad_norm=1.0, **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not isinstance(schedule, _LRSchedule) and schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        # initialize schedule object
        if not isinstance(schedule, _LRSchedule):
            schedule_type = SCHEDULES[schedule]
            schedule = schedule_type(warmup=warmup, t_total=t_total)
        else:
            if warmup != -1 or t_total != -1:
                logger.warning("warmup and t_total on the optimizer are ineffective when _LRSchedule object is provided as schedule. "
                               "Please specify custom warmup and t_total in _LRSchedule object.")
        defaults = dict(lr=lr, schedule=schedule,
                        betas=betas, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                lr_scheduled = group['lr']
                lr_scheduled *= group['schedule'].get_lr(state['step'])
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['betas']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                lr_scheduled = group['lr']
                lr_scheduled *= group['schedule'].get_lr(state['step'])

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss
