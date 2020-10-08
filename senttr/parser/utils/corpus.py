# -*- coding: utf-8 -*-

from collections import namedtuple


Sentence = namedtuple(typename='Sentence',
                      field_names=['ID', 'FORM', 'LEMMA', 'CPOS',
                                   'POS', 'FEATS', 'HEAD', 'DEPREL',
                                   'PHEAD', 'PDEPREL'],
                      defaults=[None]*10)


## transit = ['L', 'R', 'S','H']
def read_seq(in_file, vocab):
    lines = []
    with open(in_file, 'r') as f:
        for line in f:
            lines.append(line)
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split()
    gold_seq, arcs, seq = [], [], []
    max_read = 0
    for line in lines:
        #if max_read == 100:
        #   break
        if len(line) == 0:
            gold_seq.append({'act':seq, 'rel':arcs})
            max_read += 1
            arcs, seq = [], []
        elif len(line) == 3:
            assert line[0] == 'Shift'
            seq.append(2)
            arcs.append(0)
        elif len(line) == 1:
            assert line[0] == 'Swap'
            seq.append(3)
            arcs.append(0)
        elif len(line) == 2:
            if line[0].startswith('R'):
                assert line[0] == 'Right-Arc'
                seq.append(1)
                arcs.append(vocab.rel2id( line[1] ))
            elif line[0].startswith('L'):
                assert line[0] == 'Left-Arc'
                seq.append(0)
                arcs.append(vocab.rel2id( line[1] ))
    return gold_seq


class Corpus(object):
    ROOT = '<ROOT>'

    def __init__(self, sentences):
        super(Corpus, self).__init__()

        self.sentences = sentences
        self.ids = [i+1 for i in range(len(sentences))]

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(
            '\n'.join('\t'.join(map(str, i))
                      for i in zip(*(f for f in sentence if f))) + '\n'
            for sentence in self
        )

    def __getitem__(self, index):
        return self.sentences[index]


    @property
    def words(self):
        return [[self.ROOT] + list(sentence.FORM) for sentence in self]

    @property
    def tags(self):
        return [[self.ROOT] + list(sentence.CPOS) for sentence in self]

    @property
    def heads(self):
        #return [[0] + [0] + list(map(int, sentence.HEAD))+[0] for sentence in self]
        return [[0] + list(map(int, sentence.HEAD)) for sentence in self]

    @property
    def rels(self):
        #return [[self.ROOT] + [self.ROOT] + list(sentence.DEPREL)+[self.ROOT] for sentence in self]
        return [[self.ROOT] + list(sentence.DEPREL) for sentence in self]

    @heads.setter
    def heads(self, sequences):
        self.sentences = [sentence._replace(HEAD=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @rels.setter
    def rels(self, sequences):
        self.sentences = [sentence._replace(DEPREL=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @classmethod
    def load(cls, fname):
        start, sentences = 0, []
        with open(fname, 'r') as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(line) <= 1:
                sentence = Sentence(*zip(*[l.split() for l in lines[start:i]]))
                sentences.append(sentence)
                start = i + 1
        corpus = cls(sentences)

        return corpus

    def save(self, fname):
        with open(fname, 'w') as f:
            f.write(f"{self}\n")
