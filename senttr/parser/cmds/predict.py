# -*- coding: utf-8 -*-

from parser import Parser, Model
from parser.utils import Corpus
from parser.utils.data import TextDataset, batchify
import torch


class Predict(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--fdata', default='data/ptb/test.conllx',
                               help='Path to test dataset')
        subparser.add_argument('--fpred', default='pred.conllx',
                               help='Prediction path')
        subparser.add_argument('--modelname', default='None',
                               help='Path to trained model')
        subparser.add_argument('--mainpath', default='None',
                               help='Main path')
        return subparser

    def rearange(self, items, ids):

        indicies = []
        for id in ids:
            for i in id:
                indicies.append(i)
        indicies = sorted(range(len(indicies)), key=lambda k: indicies[k])
        items = [items[i] for i in indicies]
        return items

    def __call__(self, args):
        print("Load the model")

        modelpath = args.mainpath + args.model + args.modelname + "/model_weights"
        vocabpath = args.mainpath + args.vocab + args.modelname + "/vocab.tag"

        config = torch.load(modelpath)['config']


        vocab = torch.load(vocabpath)
        parser = Parser.load(modelpath)
        model = Model(vocab, parser, config, vocab.n_rels)

        print("Load the dataset")
        corpus = Corpus.load(args.fdata)
        dataset = TextDataset(vocab.numericalize(corpus))
        # set the data loader
        loader, ids = batchify(dataset,5*config.batch_size, config.buckets)

        print("Make predictions on the dataset")
        heads_pred, rels_pred, metric = model.predict(loader)

        print(metric)
        print(f"Save the predicted result to {args.fpred}")

        heads_pred = self.rearange(heads_pred, ids)
        rels_pred = self.rearange(rels_pred,ids)


        corpus.heads = heads_pred
        corpus.rels = rels_pred
        corpus.save(args.fpred)