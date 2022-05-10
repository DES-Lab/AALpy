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

    def __lt__(self, other):
        return len(self.prefix) < len(other.prefix)

    def __le__(self, other):
        return len(self.prefix) <= len(other.prefix)

    def __eq__(self, other):
        return self.prefix == other.prefix


def createPTA(data):
    root_node = RpniNode(False)
    for seq in data:
        curr_node = root_node
        for i, o in seq:
            if i not in curr_node.children.keys():
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

            for red_state in red:
                if self.compatible(self.merge(red_state, lex_min_blue, copy_nodes=True)):
                    self.merge(red_state, lex_min_blue)
                    print(red_state.prefix, lex_min_blue.prefix)
                    merged = True
                    print('MERGING')

            if not merged:
                print('ADDING TO RED')
                insort(red, lex_min_blue)

            blue.clear()
            for r in red:
                for c in r.children.values():
                    if c not in red:
                        blue.append(c)
            print(len(blue))

        assert sorted(red, key=lambda x: len(x.prefix)) == red
        return red

    def compatible(self, r):
        for seq in self.data:
            curr_node = r
            for i, o in seq:
                if i not in curr_node.children.keys():
                    return False  # TODO I DONT KNOW
                curr_node = curr_node.children[i]
                if curr_node.output != curr_node.output:
                    return False
        print('COMPATIBLE')
        return True

    def merge(self, r, lex_min_blue, copy_nodes=False):
        to_update = self.root_node if not copy_nodes else self.root_node.copy()
        red_node = r if not copy_nodes else r.copy()

        b_prefix = lex_min_blue.prefix
        for p in b_prefix[:-1]:
            to_update = to_update.children[p]

        to_update.children[b_prefix[-1]] = red_node
        print('FOLDING', red_node.prefix, lex_min_blue.prefix)
        self.fold(red_node, lex_min_blue)

        print(to_update.output)
        exit()
        return to_update

    def fold(self, red_node, blue_node):
        for i in blue_node.children.keys():
            if i in red_node.children.keys():
                print('  FOLD REC: ', red_node.children[i].prefix, blue_node.children[i].prefix)
                self.fold(red_node.children[i], blue_node.children[i])
            else:
                red_node.children[i] = blue_node.children[i]
                print('  FOLD Add: ', red_node.prefix, blue_node.children[i].prefix)

        print('  ENDING FOLD', red_node.prefix, blue_node.prefix)
        if red_node.output != blue_node.output:
            print('  Changing output ', red_node.prefix, blue_node.prefix,
                  f'from {red_node.output} to {blue_node.output}')
        red_node.output = blue_node.output

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
            seq.append((i, o))
        dfa_sul.post()
        data.append(seq)

    data = [[('a', False), ('a', False), ('a', True)],
            [('a', False), ('a', False), ('b', False), ('a', True)],
            [('b', False), ('b', False), ('a', True)],
            [('b', False), ('b', False), ('a', True), ('b', False), ('a', True)]]

    model = RPNI(data)
