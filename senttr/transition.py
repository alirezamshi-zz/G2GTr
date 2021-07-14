class ArcStandard(TransitionSystem):
    @classmethod
    def actions_list(self):
        return ['Shift', 'Swap','Left-Arc', 'Right-Arc']

    def _preparetransitionset(self, parserstate):
        SHIFT = self.mappings['action']['Shift']
        SWAP = self.mappings['action']['Swap']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        stack, buf, head = parserstate.stack, parserstate.buf, parserstate.head

        t = []

        if len(buf) > 0:
            t += [(SHIFT,)]

        if len(stack) > 1:
            t += [(SWAP,)]

        if len(stack) > 1:
            t += [(LEFTARC,)]

        if len(stack) > 1:
            t += [(RIGHTARC,)]

        parserstate._transitionset = t

    def advance(self, parserstate, action):
        SHIFT = self.mappings['action']['Shift']
        SWAP = self.mappings['action']['Swap']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])
        cand = parserstate.transitionset()

        if isinstance(action, int):
            a, rel = self.tuple_trans_from_int(cand, action)
        else:
            rel = action[-1]
            a = action[:-1]

        stack = parserstate.stack
        buf = parserstate.buf

        if a[0] == SHIFT:
            parserstate.stack = [buf[0]] + stack
            parserstate.buf = buf[1:]
        elif a[0] == SWAP:
            parserstate.stack = [stack[0]] + stack[2:]
            parserstate.buf = [stack[1]] + buf
        elif a[0] == LEFTARC:
            parserstate.head[stack[1]] = [stack[0], rel]
            parserstate.stack = [stack[0]] + stack[2:]
        elif a[0] == RIGHTARC:
            parserstate.head[stack[0]] = [stack[1], rel]
            parserstate.stack = stack[1:]

        self._preparetransitionset(parserstate)

    def goldtransition(self, parserstate, goldrels=None):
        SHIFT = self.mappings['action']['Shift']
        SWAP = self.mappings['action']['Swap']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head
        proj_order = parserstate.proj_order
        POS = len(self.mappings['pos'])


        if len(stack) < 2 and len(buf) > 0:
            return (SHIFT,-1)

        stack0_done = True
        for x in buf:
            if x in goldrels[stack[0]]:
                stack0_done = False
                break
        for y in stack:
            if y in goldrels[stack[0]]:
                stack0_done=False
                break

        stack1_done=True
        for x in buf:
            if x in goldrels[stack[1]]:
                stack1_done=False
                break
        for y in stack:
           if y in goldrels[stack[1]]:
                stack1_done=False
                break

        if stack[1] in goldrels[stack[0]] and stack1_done:
            rel = goldrels[stack[0]][stack[1]]
            return (LEFTARC, rel)
        elif stack[0] in goldrels[stack[1]] and stack0_done:
            rel = goldrels[stack[1]][stack[0]]
            return (RIGHTARC, rel)
        else:
            if stack[1] < stack[0] and proj_order[stack[0]] < proj_order[stack[1]]:
                return (SWAP, -1)
            else:
                return (SHIFT, -1)


    def trans_to_str(self, t, state, pos, fpos=None):
        SHIFT = self.mappings['action']['Shift']
        SWAP = self.mappings['action']['Swap']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        if t[0] == SHIFT:
            if fpos is None:
                return "Shift\t%s" % (pos[state.buf[0]])
            else:
                return "Shift\t%s\t%s" % (pos[state.buf[0]], fpos[state.buf[0]])
        elif t[0] == SWAP:
            return "Swap\t"
        elif t[0] == LEFTARC:
            return "Left-Arc\t%s" % (self.invmappings['rel'][t[1]])
        elif t[0] == RIGHTARC:
            return "Right-Arc\t%s" % (self.invmappings['rel'][t[1]])

    @classmethod
    def trans_from_line(self, line):
        if line[0] == 'Left-Arc':
            fields = { 'action':line[0], 'rel':line[1] }
        elif line[0] == 'Right-Arc':
            fields = { 'action':line[0], 'rel':line[1] }
        elif line[0] == 'Swap':
            fields = { 'action':line[0], 'pos':None }
        elif line[0] == 'Shift':
            fields = { 'action':line[0], 'pos':line[1] }
            if len(line) > 2:
                fields['fpos'] = line[2]
        else:
            raise ValueError(line[0])
        return fields

    def tuple_trans_to_int(self, cand, t):
        SHIFT = self.mappings['action']['Shift']
        SWAP = self.mappings['action']['Swap']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])

        base = 0
        if t[0] == SHIFT:
            return base

        base += 1

        if t[0] == SWAP:
            return base
        base += 1

        if t[0] == LEFTARC:
            return base + t[1]

        base += RELS

        if t[0] == RIGHTARC:
            return base + t[1]

    def tuple_trans_from_int(self, cand, action):
        SHIFT = self.mappings['action']['Shift']
        SWAP = self.mappings['action']['Swap']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        RELS = len(self.mappings['rel'])
        rel = -1

        base = 0
        if action == base:
            a = (SHIFT,)
        base += 1

        if action == base:
            a = (SWAP,)

        base += 1

        if base <= action < base + RELS:
            a = (LEFTARC,)
            rel = action - base
        base += RELS

        if base <= action < base + RELS:
            a = (RIGHTARC,)
            rel = action - base

        return a, rel
