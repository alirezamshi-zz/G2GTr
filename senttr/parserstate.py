"""
Parser state of transition-based parsers.
"""

from copy import copy
import numpy as np
class ParserState:
    def __init__(self, sentence, transsys=None, goldrels=None):
        self.stack = [0]

        self.sentence = sentence
        # sentences should already have a <ROOT> symbol as the first token
        self.buf = [i+1 for i in xrange(len(sentence)-1)]
        # head and relation labels
        self.head = [[-1, -1] for _ in xrange(len(sentence))]

        self.pos = [-1 for _ in xrange(len(sentence))]

        self.goldrels = goldrels

        self.proj_order = self.build_inorder()

        self.transsys = transsys
        if self.transsys is not None:
            self.transsys._preparetransitionset(self)

    def build_inorder(self):
        ids = self.stack + self.buf
        parent_ids = [-1] * len(ids)
        for i,parent in enumerate(self.goldrels):
            childs = parent.keys()
            for child in childs:
                assert parent_ids[child] == -1
                parent_ids[child] = i
        both = zip(ids,parent_ids)
        sentence = [ConllEntry(i,parent_id) for i,parent_id in both]
        assert len(sentence) == len(ids) == len(parent_ids)

        new_tokens = inorder(sentence)

        assert len(new_tokens) == len(sentence), 'before inorder,sent:{},inoder:{},parent_ids:{}' \
                                                 ''.format(len(sentence),len(new_tokens),parent_ids)
        all_ids = []
        for x in new_tokens:
            all_ids.append(x.id)

        all_ids = np.asarray(all_ids)
        all_ids = np.argsort(all_ids)

        assert len(all_ids) == len(sentence),'after inorder'

        return all_ids

    def transitionset(self):
        return self._transitionset

    def clone(self):
        res = ParserState([])
        res.stack = copy(self.stack)
        res.buf = copy(self.buf)
        res.head = copy(self.head)
        res.pos = copy(self.pos)
        res.goldrels = copy(self.goldrels)
        res.transsys = self.transsys
        if hasattr(self, '_transitionset'):
            res._transitionset = copy(self._transitionset)
        return res

class ConllEntry:
    def __init__(self, id, parent_id=None):
        self.id = id
        self.parent_id = parent_id

def inorder(sentence):
    queue = [sentence[0]]
    def inorder_helper(sentence,i):
        results = []
        left_children = [entry for entry in sentence[:i] if entry.parent_id == i]
        for child in left_children:
            results += inorder_helper(sentence,child.id)
        results.append(sentence[i])

        right_children = [entry for entry in sentence[i:] if entry.parent_id == i ]
        for child in right_children:
            results += inorder_helper(sentence,child.id)
        return results
    return inorder_helper(sentence,queue[0].id)


