import random
from bisect import insort
from copy import deepcopy

from aalpy.SULs import DfaSUL
from aalpy.utils import get_Angluin_dfa


class RpniNode:
    def __init__(self, output):
        self.output = output
        self.children = dict()
        self.prefix = ()

    def copy(self):
        return deepcopy(self)


def createPTA(data):
    root_node = RpniNode(data[0][0])
    for seq in data:
        curr_node = root_node
        for io in seq:
            i, o = io[0], io[1]
            if io not in curr_node.children.keys():
                node = RpniNode(o)
                node.prefix = curr_node.prefix + (i,)
                curr_node.children[i] = node
            curr_node = curr_node.children[i]

    return root_node


class RPNI:
    def __init__(self, data):
        self.data = data
        self.root_node = createPTA(data)
        self.run_rpni()

    def run_rpni(self):

        red = [self.root_node]
        blue = list(red[0].children.values())

        while blue:
            lex_min_blue = min(list(blue), key=lambda x: len(x.prefix))
            merged = False

            for r in red:
                if self.compatible(self.merge(r, lex_min_blue, in_compatability=True)):
                    self.merge(r, lex_min_blue)
                    merged = True
                    print('MERGING')
                    break

            if not merged:
                print('ADDING TO RED')
                insort(red, lex_min_blue)

            blue.clear()
            for r in red:
                for c in r.children.values():
                    if c not in red:
                        blue.append(c)

        assert sorted(red, key=lambda x: len(x.prefix)) == red
        return red

    def compatible(self, r):
        for seq in self.data:
            curr_node = r
            for io in seq:
                i, o = io[0], io[1]
                if i not in curr_node.children.keys():
                    break
                curr_node = curr_node.children[i]
                if curr_node.output != curr_node.output:
                    return False
        return True

    def merge(self, r, lex_min_blue, in_compatability=False):
        to_update = self.root_node if not in_compatability else self.root_node.copy()
        b_prefix = lex_min_blue.prefix
        for p in b_prefix[:-1]:
            to_update = to_update.children[p]

        to_update.children[b_prefix[-1]] = r

        self.fold(r, lex_min_blue)

        return to_update

    def fold(self, red_node, blue_node):
        for i, c in blue_node.copy().children.items():
            if i in red_node.children.keys():
                self.fold(red_node.children[i], blue_node.children[i])
            else:
                red_node.children[i] = blue_node.children[i]


if __name__ == '__main__':
    dfa = get_Angluin_dfa()
    dfa_sul = DfaSUL(dfa)
    input_al = dfa.get_input_alphabet()
    data = []
    for _ in range(10):
        dfa_sul.pre()
        seq = []
        for _ in range(5, 12):
            i = random.choice(input_al)
            o = dfa_sul.step(i)
            seq.append((i,o))
        dfa_sul.post()
        data.append(seq)

    model = RPNI(data)