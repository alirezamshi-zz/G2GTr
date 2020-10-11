#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import pickle
import time
from datetime import datetime
import argparse
import torch
from model import ParserModel
from torch import nn, optim
from tqdm import tqdm
from featurize import AverageMeter, load_and_preprocess_data_test
from utils import batch_dev_test
import numpy as np
from run import validate

if __name__ == "__main__":
    
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--datapath', default='./data_new2',
                        help='Data directory for train/test')
    
    parser.add_argument('--testfile', default='test.conll',
                        help='File to test the model')
    
    parser.add_argument('--model_name',
                        help='Model directory')

    parser.add_argument('--batchsize', default=2, type=int,
                        help='Batch size number')

    parser.add_argument('--mainpath', default='',
                        help='File to test the model')
    opt2 = parser.parse_args()       
    print(opt2)
    
    
    checkpoint = torch.load(opt2.mainpath+'/output/'+str(opt2.model_name)+'model.weights')
    
    opt = checkpoint['opt']
    
    opt.datapath = opt2.datapath
    opt.testfile = opt2.testfile
    opt.batchsize = opt2.batchsize
    opt.mainpath = opt2.mainpath


    with open(opt2.mainpath+'/vocab/'+str(opt2.model_name)+'.pkl', 'rb') as f:
        parser = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    start = time.time()
    debug = False
    test_data,pad_action = load_and_preprocess_data_test(opt,parser,debug)
    
    test_batched = batch_dev_test(test_data, opt.batchsize, parser.NULL, parser.P_NULL,
                                  parser ,no_sort=False)

    model = ParserModel(parser.embedding_shape, device, parser, pad_action, opt)
    
    model.load_state_dict(checkpoint['model'],strict=False)
    
    model = model.to(device)
    print("took {:.2f} seconds\n".format(time.time() - start))
    
    print(80 * "=")
    print("TESTING")
    print(80 * "=")
    print("Final evaluation on test set", )
    model.eval()

    UAS, LAS = validate(model, parser, test_batched, test_data, device, opt.batchsize,pad_action['P'],opt)
    print("- test UAS: {:.2f}".format(UAS * 100.0))
    print("- test LAS: {:.2f}".format(LAS * 100.0))
    print("Done!")

