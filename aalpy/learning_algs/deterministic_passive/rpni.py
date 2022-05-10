import random
from bisect import insort
from copy import deepcopy

from aalpy.SULs import DfaSUL
from aalpy.automata import DfaState, Dfa
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


def visualize_pta(rootNode):
    from pydot import Dot, Node, Edge
    graph = Dot('fpta', graph_type='digraph')

    graph.add_node(Node(str(rootNode.prefix), label=f'{rootNode.output}'))

    queue = [rootNode]
    visitied = set()
    visitied.add(rootNode.prefix)
    while queue:
        curr = queue.pop(0)
        for i, c in curr.children.items():
            if c.prefix not in visitied:
                graph.add_node(Node(str(c.prefix), label=f'{c.output}'))
            graph.add_edge(Edge(str(curr.prefix), str(c.prefix), label=f'{i}'))
            if c.prefix not in visitied:
                queue.append(c)
            visitied.add(c.prefix)

    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', str(rootNode.prefix), label=''))

    graph.write(path=f'pta.pdf', format='pdf')


class RPNI:
    def __init__(self, data):
        self.data = data
        self.root_node = createPTA(data)

    def run_rpni(self):

        red = [self.root_node]
        blue = list(red[0].children.values())

        while blue:
            lex_min_blue = min(list(blue), key=lambda x: len(x.prefix))
            merged = False

            for red_state in red:
                if self.compatible(self.merge(red_state, lex_min_blue, copy_nodes=True)):
                    self.merge(red_state, lex_min_blue)
                    print('MERGING', red_state.prefix, lex_min_blue.prefix)
                    merged = True

            if not merged:
                print('ADDING TO RED', lex_min_blue.prefix)
                insort(red, lex_min_blue)

            blue.clear()
            for r in red:
                for c in r.children.values():
                    if c not in red:
                        blue.append(c)

        assert sorted(red, key=lambda x: len(x.prefix)) == red
        print(f'RPNI returned {len(red)} state automaton')
        return self.to_automaton(red)

    def compatible(self, r):
        for sequance in self.data:
            curr_node = r
            for i, o in sequance:
                if i not in curr_node.children.keys():
                    return False  # TODO I DON'T KNOW
                curr_node = curr_node.children[i]
                if curr_node.output != o:
                    return False
        return True

    def merge(self, r, lex_min_blue, copy_nodes=False):
        root_node = self.root_node.copy() if copy_nodes else self.root_node
        lex_min_blue = lex_min_blue.copy() if copy_nodes else lex_min_blue

        red_node = root_node
        for p in r.prefix:
            red_node = red_node.children[p]

        b_prefix = lex_min_blue.prefix
        to_update = root_node
        for p in b_prefix[:-1]:
            to_update = to_update.children[p]

        to_update.children[b_prefix[-1]] = red_node
        self.fold(red_node, lex_min_blue)

        return root_node

    def fold(self, red_node, blue_node):
        red_node.output = blue_node.output
        for i in blue_node.children.keys():
            if i in red_node.children.keys():
                self.fold(red_node.children[i], blue_node.children[i])
            else:
                red_node.children[i] = blue_node.children[i].copy()

    def to_automaton(self, red):
        state, automaton = DfaState, Dfa

        initial_state = None
        prefix_state_map = {}
        for i, r in enumerate(red):
            prefix_state_map[r.prefix] = DfaState(state_id=f's{i}', is_accepting=r.output)
            if i == 0:
                initial_state = prefix_state_map[r.prefix]

        for r in red:
            for i, c in r.children.items():
                prefix_state_map[r.prefix].transitions[i] = prefix_state_map[c.prefix]
        return Dfa(initial_state, list(prefix_state_map.values()))


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

    # data = [[('a', False), ('a', False), ('a', True)],
    #         [('a', False), ('a', False), ('b', False), ('a', True)],
    #         [('b', False), ('b', False), ('a', True)],
    #         [('b', False), ('b', False), ('a', True), ('b', False), ('a', True)],
    #         [('a', False,), ('b', False,), ('a', False)]]
    # a,bb,aab,aba
    model = RPNI(data).run_rpni()
    print(model)
