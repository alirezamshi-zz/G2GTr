#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import math
import os
from os import path
import pickle
import time
from datetime import datetime
import argparse
import torch
from model import ParserModel
from torch import nn, optim
from tqdm import tqdm
from featurize import AverageMeter, load_and_preprocess_data,punct,load_and_preprocess_datap
import numpy as np
import operator
import torch.cuda as cuda
from utils import update_state,batch_train,batch_dev_test,\
    prepare_graph,batched_index_select,prepare_data
from utils import BertAdam
from transformers import get_linear_schedule_with_warmup

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'


def validate(model,
             parser,
             dev_batched,
             dataset,
             device,
             batch_size_eval,
             pad_action,
             opt):

    batch_del_label = None
    graph_emb = None
    
    n_batchs = len(dev_batched)
    dependencies_total = []
    pbar = tqdm(total= n_batchs)
    for batch in range(n_batchs):

        ################################## preparing the data ##################################
        batch_buffer_ind, batch_buffer, batch_buffer_pos, mask_buffer = dev_batched[batch]

        batch_buffer = batch_buffer.to(device)
        batch_buffer_ind = batch_buffer_ind.to(device)
        batch_buffer_pos = batch_buffer_pos.to(device)
        mask_buffer = mask_buffer.to(device)
        batch_size = batch_buffer.size()[0]

        if opt.graphinput:
            batch_delete = batch_buffer.clone()
            batch_delete_pos = batch_buffer_pos.clone()

            graph_emb = torch.zeros((batch_size,batch_buffer.size()[1]+2,
                                     batch_buffer.size()[1])).long().to(device)
            graph_input = torch.zeros((batch_size,3*batch_buffer.size()[1]+4,
                                       3*batch_buffer.size()[1]+4)).long().to(device)


        #### build attention mask
        mask_stack = torch.zeros((batch_size,batch_buffer.size()[1]+1)).byte().to(device)
        mask_stack[:,-1] = 1
        
        #### build stack word
        batch_stack = torch.zeros((batch_size,batch_buffer.size()[1]+1)).long().to(device)
        batch_stack[:,-1] = parser.ROOT
        
        batch_stack_ind = ( torch.ones((batch_size,batch_buffer.size()[1]+1)) *
                            (batch_buffer.size()[1]+1) ).long().to(device)
        batch_stack_ind[:,-1] = 0

        #### build stack POS
        batch_stack_pos = torch.zeros((batch_size,batch_buffer.size()[1]+1)).long().to(device)
        batch_stack_pos[:,-1] = parser.P_ROOT
        
        #### build dep
        batch_dep = torch.ones_like(batch_stack) * parser.NULL
        batch_dep_pos = torch.ones_like(batch_stack_pos) * parser.P_NULL
        batch_dep_label = torch.ones_like(batch_stack_pos) * parser.L_NULL

        batch_dep_buffer = torch.ones_like(batch_stack[:, :-1]) * parser.NULL
        batch_dep_pos_buffer = torch.ones_like(batch_stack_pos[:, :-1]) * parser.P_NULL
        batch_dep_label_buffer = torch.ones_like(batch_stack_pos[:, :-1]) * parser.L_NULL
        
        ### token_type_ids
        if opt.graphinput:
            token_type_ids = torch.zeros((batch_size,3*batch_buffer.size()[1]+4)).long().to(device)
            token_type_ids[:,:batch_buffer.size()[1]+2] = 2
            token_type_ids[:,batch_buffer.size()[1]+2:2*batch_buffer.size()[1]+3] = 1
        else:
            token_type_ids = torch.zeros((batch_size,2*batch_buffer.size()[1]+4)).long().to(device)
            token_type_ids[:,:batch_buffer.size()[1]+3] = 1
            
        
        if opt.seppoint:
            sep_point = batch_buffer.size()[1]+1
        else:
            sep_point = 0
        
        ### build the dependency set
        dependencies = []
        for _ in range(batch_size):
            dependencies.append([])
                
        ###### do the initialization of model
        batch_size = len(batch_buffer)
        
        mask_cls = torch.ones((batch_size,1)).byte().to(device)
        mask_sep = torch.ones((batch_size,1)).byte().to(device)

        step_i = 0
        
        ## clip cls and sep 
        ones = (torch.ones((batch_size,1)) ).long().to(device)
        batch_temp = batch_dep[:,:-1].clone()
        batch_pos_temp = batch_dep_pos[:,:-1].clone()
        batch_label_temp = batch_dep_label[:,:-1].clone()
        if opt.graphinput:
            batch_del_label = batch_label_temp.clone()
        
        update = torch.zeros(batch_size).long().to(device)
        action_state = None
        action_cell = None
        transitions = None
        labels = None

        while True:
            if len(torch.nonzero(update)) == batch_size:
                break
              
            if opt.graphinput:
                mask_delete,graph_in = prepare_graph(graph_emb,batch_stack_ind,
                                                     batch_buffer_ind,batch_size,graph_input,device,parser.NULL)
                input_ids_x, pos_ids_x, batch_dep_input, batch_dep_pos_input,batch_dep_label_input,\
                batch_graph_label_input, attention_mask = prepare_data(ones,opt.graphinput,parser,
                        batch_stack,batch_buffer,batch_stack_pos,batch_buffer_pos,
                        batch_dep,batch_temp,batch_dep_pos,batch_pos_temp,batch_dep_label,batch_label_temp, mask_cls,
                        mask_stack,mask_sep,mask_buffer,batch_dep_buffer,batch_dep_pos_buffer,batch_dep_label_buffer,
                        mask_delete,batch_del_label,batch_delete,batch_delete_pos)
                transitions, labels, action_state, action_cell = model(1,sep_point,input_ids_x,pos_ids_x,batch_dep_input,
                                                                        batch_dep_pos_input,batch_dep_label_input,
                                                                        batch_graph_label_input,attention_mask,update,
                                                                        token_type_ids,mask_stack,mask_buffer,batch_stack_ind,
                                                                        transitions,labels,action_state,action_cell,graph_in)
            
            else:    
                input_ids_x, pos_ids_x, batch_dep_input, batch_dep_pos_input,batch_dep_label_input,attention_mask =\
                prepare_data(ones,opt.graphinput,parser,batch_stack,batch_buffer,batch_stack_pos,batch_buffer_pos,
                            batch_dep,batch_temp,batch_dep_pos,batch_pos_temp,batch_dep_label,batch_label_temp,
                            mask_cls,mask_stack,mask_sep,mask_buffer,batch_dep_buffer,batch_dep_pos_buffer,
                            batch_dep_label_buffer)
                
                    
                transitions, labels, action_state, action_cell = model(1,sep_point,input_ids_x,pos_ids_x,batch_dep_input,
                                                                        batch_dep_pos_input,batch_dep_label_input,None,
                                                                        attention_mask,update,token_type_ids,mask_stack,
                                                                        mask_buffer,batch_stack_ind,transitions,labels,
                                                                        action_state,action_cell)

            ####### update stack and buffer ##############################
            batch_stack_ind,batch_stack,batch_buffer_ind,batch_buffer,batch_stack_pos,batch_buffer_pos,\
            batch_dep,batch_dep_pos,mask_buffer,mask_stack, batch_dep_label, graph_emb,batch_del_label,\
            batch_dep_buffer,batch_dep_pos_buffer,batch_dep_label_buffer,\
            dependencies = update_state(1,opt,batch_stack_ind, batch_stack, batch_buffer_ind,
                        batch_buffer, batch_stack_pos,batch_buffer_pos,batch_dep,batch_dep_pos,mask_buffer,
                        mask_stack, transitions, batch_dep_label,labels,parser.NULL,parser.P_NULL,parser.L_NULL,
                        batch_dep_buffer,batch_dep_pos_buffer,batch_dep_label_buffer,
                        graph_emb,batch_del_label,dependencies=dependencies)

            update = ((mask_buffer.sum(dim=1)==0) * (mask_stack.sum(dim=1)==1) * pad_action * 1.0).long()
            
            del input_ids_x, pos_ids_x, batch_dep_input,\
                batch_dep_pos_input,batch_dep_label_input,attention_mask
            if opt.graphinput:
                del mask_delete,graph_in
            ######################################################################################
            
            step_i += 1
                
                
        pbar.update(1)    
        del batch_buffer_ind, batch_buffer, batch_buffer_pos,mask_buffer, \
                batch_stack_ind, batch_stack, batch_stack_pos, token_type_ids, batch_dep, batch_dep_pos,ones,\
                batch_label_temp,batch_dep_label,batch_dep_buffer,batch_dep_pos_buffer,batch_dep_label_buffer
        if opt.graphinput:
            del batch_delete,graph_emb,graph_input,batch_del_label,batch_delete_pos

        dependencies_total.extend(dependencies)
        
    with open(opt.mainpath+'/dependency/'+str(opt.outputname)+'.pkl', 'wb') as f:
        pickle.dump(dependencies_total, f, pickle.HIGHEST_PROTOCOL) 
        
        
    UAS = all_tokens = 0.0
    LAS = 0.0
    with tqdm(total=len(dataset)) as prog:
        for i, ex in enumerate(dataset):
            head = [-1] * len(ex['word'])
            label = [-1] * len(ex['word'])
            for h, t, l in dependencies_total[i]:
                head[t] = h
                label[t] = l
            for pred_h, pred_l, gold_h, gold_l, pos in zip(
                    head[1:], label[1:], ex['head'][1:], ex['label'][1:],
                    ex['pos'][1:]):
                
                assert parser.id2tok[pos].startswith(P_PREFIX)
                pos_str = parser.id2tok[pos][len(P_PREFIX):]
                UAS += 1.0 if pred_h == gold_h else 0.0
                if pred_h == gold_h and pred_l == gold_l:
                    LAS += 1.0
                all_tokens += 1
                    
            prog.update(i + 1)
    UAS /= all_tokens
    LAS /= all_tokens
    del dependencies_total
    
    return UAS,LAS
    

def adjust_learning_rate(lr,update_lr, optimizer, epoch):
    lr = lr * (0.9 ** (epoch // update_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model,
          parser,
          train_batched,
          train_data,
          dev_batched,
          dev_data_total,
          output_path,
          device,
          max_seq_length,
          mean_seq_length,
          emb_size,
          opt,
          pad_action):

    pad_action = pad_action['P']
    
    best_dev_LAS = 0
    n_batchs = len(train_batched)
    
    if opt.mean_seq:
        num_train_optimization_steps = opt.nepochs * mean_seq_length * n_batchs
    else:
        num_train_optimization_steps = opt.nepochs * max_seq_length * n_batchs
        
    print('number of steps')
    print(num_train_optimization_steps)
    ## define the optimizer
    if opt.Bertoptim:
        if opt.use_two_opts:
            print("use two optimizers")
            model_nonbert = []
            model_bert = []
            layernorm_params = ['LayerNormKeys', 'dp_relation_k', 'dp_relation_v',
                         'compose','label_emb','pooler.dense','pooler.dense_label']
            for name, param in model.named_parameters():
                if 'bertmodel' in name and not any(nd in name for nd in layernorm_params):
                    model_bert.append(param)
                else:
                    model_nonbert.append(param)
            optimizer = BertAdam(model_bert,
                             lr=opt.lr,
                             warmup=opt.warmupproportion,
                             t_total=num_train_optimization_steps)
            optimizer_nonbert = BertAdam(model_nonbert,
                             lr=opt.lr_nonbert,
                             warmup=opt.warmupproportion,
                             t_total=num_train_optimization_steps)
        else:
            optimizer = BertAdam(model.parameters(),
                             lr=opt.lr,
                             warmup=opt.warmupproportion,
                             t_total=num_train_optimization_steps)
    else:    
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    ## load optimizer state from a checkpoint
    if opt.pretrained:
        state_dict = torch.load(opt.mainpath+'/output/' +
                                str(opt.modelpath)+"model.weights"+"pretrained")
        optimizer.load_state_dict(state_dict['optimizer'])
        del state_dict

    loss_func = nn.CrossEntropyLoss() 
    loss_func_label = nn.CrossEntropyLoss() 
    
    batch_del_label = None
    graph_emb = None
    
    n_epochs = opt.nepochs - opt.real_epoch

    for n_epoch in range(n_epochs):

        print("Epoch {:} out of {:}".format(n_epoch + 1, opt.nepochs))
        model.train()
        if not opt.Bertoptim:
            adjust_learning_rate(opt.lr,opt.updatelr, optimizer, n_epoch)
        loss_meter = AverageMeter()
        iters = np.arange(n_batchs)
        if opt.shuffle:
            print('Do shuffling')
            random.shuffle(iters)
        loss = 0.
        pbar = tqdm(total= n_batchs)
        for it in iters: 
        
            ################################## preparing the data #################################
            batch_buffer_ind,batch_buffer, batch_buffer_pos, mask_buffer ,\
                actions_batch, actions_mask_batch, batch_labels, mask_labels = train_batched[it]

            actions_batch = actions_batch.to(device)
            actions_mask_batch = actions_mask_batch.to(device)
            batch_labels = batch_labels.to(device)
            mask_labels = mask_labels.to(device)
            
            batch_buffer = batch_buffer.to(device)
            batch_buffer_ind = batch_buffer_ind.to(device)
      
            batch_buffer_pos = batch_buffer_pos.to(device)
            mask_buffer = mask_buffer.to(device)
            
            batch_size = len(batch_buffer)

            if opt.graphinput:
                batch_delete = batch_buffer.clone()
                batch_delete_pos = batch_buffer_pos.clone()

                graph_emb = torch.zeros((batch_size,batch_buffer.size()[1]+2,
                                         batch_buffer.size()[1])).long().to(device)
                graph_input = torch.zeros((batch_size,3*batch_buffer.size()[1]+4,
                                           3*batch_buffer.size()[1]+4)).long().to(device)

            #### build attention mask
            mask_stack = torch.zeros((batch_size,batch_buffer.size()[1]+1)).byte().to(device)
            mask_stack[:,-1] = 1
        
            #### build stack word
            batch_stack = torch.zeros((batch_size,batch_buffer.size()[1]+1)).long().to(device)
            batch_stack[:,-1] = parser.ROOT
            batch_stack_ind = (torch.ones((batch_size,batch_buffer.size()[1]+1)) *
                               (batch_buffer.size()[1]+1) ).long().to(device)
            batch_stack_ind[:,-1] = 0
            #### build stack POS
            batch_stack_pos = torch.zeros((batch_size,batch_buffer.size()[1]+1)).long().to(device)
            batch_stack_pos[:,-1] = parser.P_ROOT
        
            #### build dep
            batch_dep = torch.ones_like(batch_stack) * parser.NULL
            batch_dep_pos = torch.ones_like(batch_stack_pos) * parser.P_NULL
            batch_dep_label = torch.ones_like(batch_stack_pos) * parser.L_NULL

            batch_dep_buffer = torch.ones_like(batch_stack[:,:-1]) * parser.NULL
            batch_dep_pos_buffer = torch.ones_like(batch_stack_pos[:,:-1]) * parser.P_NULL
            batch_dep_label_buffer = torch.ones_like(batch_stack_pos[:,:-1]) * parser.L_NULL
            
            ### token_type_ids
            if opt.graphinput:
                token_type_ids = torch.zeros((batch_size,3*batch_buffer.size()[1]+4)).long().to(device)
                token_type_ids[:,:batch_buffer.size()[1]+2] = 2
                token_type_ids[:,batch_buffer.size()[1]+2:2*batch_buffer.size()[1]+3] = 1
            else:
                token_type_ids = torch.zeros((batch_size,2*batch_buffer.size()[1]+4)).long().to(device)
                token_type_ids[:,:batch_buffer.size()[1]+3] = 1
            
            if opt.seppoint:
                sep_point = batch_buffer.size()[1]+1
            else:
                sep_point = 0
            mask_cls = torch.ones((batch_size,1)).byte().to(device)
            mask_sep = torch.ones((batch_size,1)).byte().to(device)
            
            # main loop
            if actions_batch is None:
                step_length = opt.maxsteplength
            else:
                step_length = actions_batch.size()[1]
                
            step_i = 0
            ## clip cls and sep 
            ones = (torch.ones((batch_size,1))).long().to(device)
            batch_temp = batch_dep[:,:-1].clone()
            batch_pos_temp = batch_dep_pos[:,:-1].clone()
            batch_label_temp = batch_dep_label[:,:-1].clone()
            
            if opt.graphinput:
                batch_del_label = batch_label_temp.clone()
            
            update = torch.zeros(batch_size).long().to(device)
            
            action_state = None
            action_cell = None

            mode = 0
            while True:
                if step_i == step_length - 1:
                    mode = 1
                if len(torch.nonzero(update)) == batch_size:
                    break
                    
                if step_i == 0:
                    prev_transitions = None
                else:
                    prev_transitions = actions_batch[:,step_i-1]

                if step_i == 0:
                    prev_labels = None
                else:
                    prev_labels = batch_labels[:,step_i-1]

                transitions = actions_batch[:,step_i]
                steps = batch_labels[:,step_i]

                if opt.graphinput:
                    ## build graoh input matrix
                    mask_delete,graph_in = prepare_graph(graph_emb,batch_stack_ind,batch_buffer_ind,
                                                         batch_size,graph_input,device,parser.NULL)
                    ## prepare data for transformer
                    input_ids_x, pos_ids_x, batch_dep_input, batch_dep_pos_input,batch_dep_label_input,\
                    batch_graph_label_input,attention_mask =prepare_data(ones,opt.graphinput,parser,batch_stack,
                            batch_buffer,batch_stack_pos,batch_buffer_pos,batch_dep,batch_temp,batch_dep_pos,
                            batch_pos_temp,batch_dep_label,batch_label_temp, mask_cls,mask_stack,mask_sep,mask_buffer,
                            batch_dep_buffer,batch_dep_pos_buffer,batch_dep_label_buffer,mask_delete,batch_del_label
                                                                         ,batch_delete,batch_delete_pos)

                    output_batch,output_label,action_state,action_cell = model(0,sep_point,input_ids_x,pos_ids_x,
                                                                            batch_dep_input,batch_dep_pos_input,
                                                                            batch_dep_label_input,batch_graph_label_input,
                                                                            attention_mask,update,token_type_ids,mask_stack,
                                                                            mask_buffer,batch_stack_ind,prev_transitions,
                                                                            prev_labels,action_state,action_cell,graph_in)
                else:
                    input_ids_x, pos_ids_x, batch_dep_input, batch_dep_pos_input,batch_dep_label_input,attention_mask =\
                    prepare_data(ones,opt.graphinput,parser,batch_stack,batch_buffer,batch_stack_pos,batch_buffer_pos,
                            batch_dep,batch_temp,batch_dep_pos,batch_pos_temp,batch_dep_label,batch_label_temp,
                            mask_cls,mask_stack,mask_sep,mask_buffer,batch_dep_buffer,batch_dep_pos_buffer,
                            batch_dep_label_buffer)
                
                    output_batch,output_label,action_state,action_cell = model(0,sep_point,input_ids_x,pos_ids_x,
                                                                            batch_dep_input,batch_dep_pos_input,
                                                                            batch_dep_label_input,None,
                                                                            attention_mask,update,token_type_ids,mask_stack,
                                                                            mask_buffer,batch_stack_ind,prev_transitions,
                                                                            prev_labels,action_state,action_cell)
                ##################### mask the output batch #######################################
                action_batch = actions_batch[:,step_i]
                action_mask_batch = actions_mask_batch[:,step_i]
                
                action_batch = action_batch.masked_select(action_mask_batch)
                output_batch = output_batch.masked_select(action_mask_batch.unsqueeze(0).t()).view(-1,opt.nclass)

                ############################ mask the output label batch ###########################
                batch_label = batch_labels[:,step_i]
                mask_label = mask_labels[:,step_i]
                
                batch_label = batch_label.masked_select(mask_label)
                output_label = output_label.masked_select(mask_label.unsqueeze(0).t()).view(-1,parser.n_transit-1)

                if output_label.nelement():
                    loss = loss_func(output_batch, action_batch) + loss_func_label(output_label,batch_label)
                else:
                    loss = loss_func(output_batch, action_batch)

                ## back propagation
                if opt.use_two_opts:
                    optimizer_nonbert.zero_grad()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                if mode:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)

                if opt.use_two_opts:
                    optimizer.step()
                    optimizer_nonbert.step()
                else:
                    optimizer.step()
            
                loss_meter.update(loss.item())
            
            
                del output_batch,output_label,input_ids_x, pos_ids_x, batch_dep_input,\
                    batch_dep_pos_input,batch_dep_label_input,attention_mask
                if opt.graphinput:
                    del mask_delete,graph_in
                ############################## update stack and buffer ##############################
                batch_stack_ind,batch_stack,batch_buffer_ind,batch_buffer,batch_stack_pos,batch_buffer_pos,\
                batch_dep,batch_dep_pos,mask_buffer,mask_stack, batch_dep_label, graph_emb,batch_del_label,\
                batch_dep_buffer,batch_dep_pos_buffer,batch_dep_label_buffer = update_state(0,opt,batch_stack_ind,
                            batch_stack, batch_buffer_ind, batch_buffer,batch_stack_pos, batch_buffer_pos,batch_dep,
                            batch_dep_pos,mask_buffer,mask_stack,transitions, batch_dep_label,steps,parser.NULL,
                            parser.P_NULL,parser.L_NULL,batch_dep_buffer,batch_dep_pos_buffer,batch_dep_label_buffer,
                            graph_emb,batch_del_label)
                update = ((mask_buffer.sum(dim=1)==0) * (mask_stack.sum(dim=1)==1) * pad_action).long()
                step_i += 1

            pbar.update(1)    
            del actions_batch,batch_labels,mask_labels,batch_buffer, batch_buffer_pos,mask_buffer, actions_mask_batch, \
                    batch_stack, batch_stack_pos, token_type_ids, batch_dep, batch_dep_pos,batch_dep_label,\
                    ones, batch_dep_buffer,batch_dep_pos_buffer,batch_dep_label_buffer
            if opt.graphinput:
                del batch_delete,batch_delete_pos,graph_emb,batch_del_label,graph_input
        
        print("Average Train Loss: {}".format(loss_meter.avg))
        print("")  

        print("Evaluating on dev set", )
        
        model.eval()
        
        dev_UAS,dev_LAS = validate(model,parser,dev_batched,dev_data_total,device,opt.batchsize,pad_action,opt)
        print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
        print("- dev LAS: {:.2f}".format(dev_LAS * 100.0))
        if dev_LAS > best_dev_LAS:
            best_dev_LAS = dev_LAS
            print("New best dev UAS! Saving model.")
            torch.save({'model': model.state_dict(), 'opt':opt, 'optimizer':optimizer.state_dict() }, output_path)
            
        torch.save({'model': model.state_dict(), 'opt':opt, 'optimizer':optimizer.state_dict() }, output_path+"pretrained")
            
if __name__ == "__main__":
    
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--withpunct', default=False,action='store_true',
                        help='Use punctuation')

    parser.add_argument('--graphinput', default=False,action='store_true',
                        help='Input graph to the model')
    
    parser.add_argument('--poolinghid', default=False,action='store_true',
                        help='Max Pooling the last hidden layer instead of CLS')
    
    parser.add_argument('--unlabeled', default=False,action='store_true',
                        help='Unlabeled dependency parsing')

    parser.add_argument('--freezedp', default=False,action='store_true',
                        help='Freeze the dependency relation embeddings')
    
    parser.add_argument('--lowercase',default=False,action='store_true',
                        help='Lowercase the words')
    
    parser.add_argument('--usepos', default=False,action='store_true',
                        help='Use POS tagger')
    
    parser.add_argument('--Bertoptim', default=False,action='store_true',
                        help='Use BertAdam for optimization')

    parser.add_argument('--pretrained', default=False,action='store_true',
                        help='Start with a checkpoint')

    parser.add_argument('--withbert', default=False,action='store_true',
                        help='Initialize the model with BERT')

    parser.add_argument('--bertname', default='bert-mult-cased',
                        help='Type of Pre-trained BERT')

    parser.add_argument('--bertpath', default='',
                        help='Type of Pre-trained BERT')

    parser.add_argument('--fhistmodel', default=False,action='store_true',
                        help='Apply history model')
    
    parser.add_argument('--fcompmodel', default=False,action='store_true',
                        help='Apply composition model')

    parser.add_argument('--layernorm', default=False,action='store_true',
                        help='Layer normalization for graph input')

    parser.add_argument('--multigpu', default=False,action='store_true',
                        help='Run the model on multiple GPUs') 
    
    parser.add_argument('--seppoint', default=False,action='store_true',
                        help='Use CLS for dependency classifiers or graph output mechanism')

    parser.add_argument('--mean_seq', default=False,action='store_true',
                        help='Used for computing total number of steps')
    
    parser.add_argument('--language', default='english',
                        help='Language to train')
    
    parser.add_argument('--datapath', default='./data_new2',
                        help='Data directory for train/test')
    
    parser.add_argument('--trainfile', default='train.conll',
                        help='File to train the model')
    
    parser.add_argument('--devfile', default='dev.conll',
                        help='File to validate the model')
    
    parser.add_argument('--testfile', default='test.conll',
                        help='File to test the model')

    parser.add_argument('--seqpath', default='train.seq',
                        help='File to test the model')

    parser.add_argument('--outputname',
                        help='Name of the output model')

    parser.add_argument('--batchsize', default=2, type=int,
                        help='Batch size number')
    
    parser.add_argument('--nepochs', default=2, type=int,
                        help='Number of epochs')

    parser.add_argument('--real_epoch', default=0, type=int,
                        help='Number of epochs that is reduced from total epochs (checkpoint)')

    parser.add_argument('--lr', default=0.00001, type=float,
                        help='Learning rate for training') 
    
    parser.add_argument('--shuffle', default=True,action='store_true',
                        help='Shuffle training inputs')
    
    parser.add_argument('--ffhidden', default=200, type=int,
                        help='Size of hidden layer in classifier')

    parser.add_argument('--clipword', default=0.99, type=float,
                        help='Percentage of keeping the orginal words of dataset')
    
    parser.add_argument('--nclass', default=4, type=int,
                        help='Number of classes in classifier')
    
    parser.add_argument('--ffdropout', default=0.05, type=float,
                        help='Amount of drop-out in classifier')
    
    parser.add_argument('--nlayershistory', default=2, type=int,
                        help='Number of layers in LSTM history model')
    
    parser.add_argument('--embsize', default=768, type=int,
                        help='Dimension of Embeddings')
    
    parser.add_argument('--maxsteplength', default=300, type=int,
                        help='Maximum size of steps to de done on validation/test time')

    parser.add_argument('--updatelr', default=1, type=int,
                        help='Step to update the learning rate')
    
    parser.add_argument('--hiddensizelabel', default=200, type=int,
                        help='Size of hidden layer in label classifier')
    
    parser.add_argument('--histsize', default=768, type=int,
                        help='Size of embedding in history model') 
    
    parser.add_argument('--labelemb', default=768, type=int,
                        help='Size of label embeddings')    

    parser.add_argument('--nattentionlayer', default=6, type=int,
                        help='Number of layers in self-attention model') 
    
    parser.add_argument('--nattentionheads', default=12, type=int,
                        help='Number of attention heads in self-attention model')   
    
    parser.add_argument('--warmupproportion', default=0.01, type=float,
                        help='Proportion of warm-up for BertAdam optimizer')
    
    parser.add_argument('--modelpath',
                        help='Name of the pretrained model')

    parser.add_argument('--use_topbuffer', default=False,action='store_true',
                        help='Use also top element of Buffer')

    parser.add_argument('--use_justexist', default=False,action='store_true',
                        help='Use top buffer just for exist classifier')

    parser.add_argument('--use_two_opts', default=False,action='store_true',
                        help='Use two optimizers for training')

    parser.add_argument('--lr_nonbert', default=1e-3, type=float,
                        help='Learning rate for non-bert')

    parser.add_argument('--mainpath', default='',
                        help='File to pre-trained char embeddings')

    parser.add_argument('--debug', default=False,action='store_true',
                        help='Debug phase')

    opt = parser.parse_args()
    print(opt)

    debug = opt.debug
    if path.exists(opt.mainpath + "/vocab") != True:
        os.mkdir(opt.mainpath + "/vocab")
    if path.exists(opt.mainpath + "/dependency") != True:
        os.mkdir(opt.mainpath + "/dependency")
    if path.exists(opt.mainpath + "/output") != True:
        os.mkdir(opt.mainpath + "/output")


    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    ## build parser,dictionary, and datasets
    if opt.pretrained:
        with open('./vocab/'+str(opt.modelpath)+'.pkl', 'rb') as f:
            parser = pickle.load(f)
        embeddings, train_data, train_set, dev_data, test_data, pad_action =\
            load_and_preprocess_datap(opt,parser,debug)
    else:
        parser, embeddings, train_data, train_set, dev_data, test_data, \
        pad_action = load_and_preprocess_data(opt,debug)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_batched,max_seq_length,mean_seq_length = batch_train(train_data, opt.batchsize, pad_action, parser)
    dev_batched = batch_dev_test(dev_data, opt.batchsize, parser.NULL, parser.P_NULL, parser)
    test_batched = batch_dev_test(test_data, opt.batchsize, parser.NULL, parser.P_NULL, parser)

    start = time.time()
    
    model = ParserModel(embeddings.shape[0], device, parser, pad_action, opt, embeddings.shape[1])
    print("number of pars:{}".format(sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)))
    if opt.pretrained:
        state_dict = torch.load(opt.mainpath+'/output/' +
                                str(opt.modelpath)+"model.weights"+"pretrained")
        model.load_state_dict(state_dict['model'])
        del state_dict
    
    if opt.multigpu:
        print('multi')
        print(torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

    model = model.to(device)
    
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_path = opt.mainpath+"/output/"+str(opt.outputname)+"model.weights"
    
    parser.embedding_shape = embeddings.shape[0]
    
    embeddings_shape = embeddings.shape[1]
    
    del embeddings

    with open(opt.mainpath+'/vocab/'+str(opt.outputname)+'.pkl', 'wb') as f:
        pickle.dump(parser, f, pickle.HIGHEST_PROTOCOL) 
        
    train(
        model,
        parser,
        train_batched,
        train_set,
        dev_batched,
        dev_data,
        output_path,
        device,
        max_seq_length,
        mean_seq_length,
        embeddings_shape,
        opt,
        pad_action)
        
    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        print("Final evaluation on test set", )
        model.eval()
        UAS,LAS = validate(model, parser, test_batched, test_data, device, opt.batchsize,
                           pad_action['P'],opt)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("- test LAS: {:.2f}".format(LAS * 100.0))        
        print("Done!")
