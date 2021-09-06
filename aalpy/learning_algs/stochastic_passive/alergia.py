import os
from copy import deepcopy
from math import sqrt, log

from pydot import Dot, Node, Edge

from aalpy.learning_algs.stochastic_passive.DataHandler import IODelimiterTokenizer


def get_fptas():
    root = FptaNode('A')
    root.children['B'] = FptaNode('B')
    root.children['B'].frequency = 15
    root.children['B'].prefix = ['B']
    root.children['A'] = FptaNode('A')
    root.children['A'].frequency = 7
    root.children['A'].prefix = ['A']

    root.children['B'].children['A'] = FptaNode('A')
    root.children['B'].children['A'].frequency = 14
    root.children['B'].children['A'].prefix = ['B', 'A']

    root.children['B'].children['A'].children['B'] = FptaNode('B')
    root.children['B'].children['A'].children['B'].frequency = 10
    root.children['B'].children['A'].children['B'].prefix = ['B', 'A', 'B']
    root.children['B'].children['A'].children['A'] = FptaNode('A')
    root.children['B'].children['A'].children['A'].frequency = 4
    root.children['B'].children['A'].children['A'].prefix = ['B', 'A', 'A']
    root.children['B'].children['A'].children['B'].children['A'] = FptaNode('A')
    root.children['B'].children['A'].children['B'].children['A'].frequency = 8
    root.children['B'].children['A'].children['B'].children['A'].prefix = ['B', 'A', 'B', 'A']

    root.children['B'].children['A'].children['A'].children['A'] = FptaNode('A')
    root.children['B'].children['A'].children['A'].children['A'].frequency = 3
    root.children['B'].children['A'].children['A'].children['A'].prefix = ['B', 'A', 'A', 'A']

    root.children['A'].children['A'] = FptaNode('A')
    root.children['A'].children['A'].frequency = 5
    root.children['A'].children['A'].prefix = ['A', 'A']
    root.children['A'].children['A'].children['A'] = FptaNode('A')
    root.children['A'].children['A'].children['A'].frequency = 5
    root.children['A'].children['A'].children['A'].prefix = ['A', 'A', 'A']

    root1 = FptaNode('A')
    root1.children['B'] = FptaNode('B')
    root1.children['B'].frequency = 15
    root1.children['B'].prefix = ['B']
    root1.children['A'] = FptaNode('A')
    root1.children['A'].frequency = 7
    root1.children['A'].prefix = ['A']

    root1.children['B'].children['A'] = FptaNode('A')
    root1.children['B'].children['A'].frequency = 14
    root1.children['B'].children['A'].prefix = ['B', 'A']

    root1.children['B'].children['A'].children['B'] = FptaNode('B')
    root1.children['B'].children['A'].children['B'].frequency = 10
    root1.children['B'].children['A'].children['B'].prefix = ['B', 'A', 'B']
    root1.children['B'].children['A'].children['A'] = FptaNode('A')
    root1.children['B'].children['A'].children['A'].frequency = 4
    root1.children['B'].children['A'].children['A'].prefix = ['B', 'A', 'A']
    root1.children['B'].children['A'].children['B'].children['A'] = FptaNode('A')
    root1.children['B'].children['A'].children['B'].children['A'].frequency = 8
    root1.children['B'].children['A'].children['B'].children['A'].prefix = ['B', 'A', 'B', 'A']

    root1.children['B'].children['A'].children['A'].children['A'] = FptaNode('A')
    root1.children['B'].children['A'].children['A'].children['A'].frequency = 3
    root1.children['B'].children['A'].children['A'].children['A'].prefix = ['B', 'A', 'A', 'A']

    root1.children['A'].children['A'] = FptaNode('A')
    root1.children['A'].children['A'].frequency = 5
    root1.children['A'].children['A'].prefix = ['A', 'A']
    root1.children['A'].children['A'].children['A'] = FptaNode('A')
    root1.children['A'].children['A'].children['A'].frequency = 5
    root1.children['A'].children['A'].children['A'].prefix = ['A', 'A', 'A']

    return root, root1

class FptaNode:

    def __init__(self, output):
        self.output = output
        self.frequency = 0
        self.children = dict()
        self.prefix = []
        # for visualization
        self.state_id = None
        self.children_prob = dict()

    def succs(self):
        return list(self.children.values())

    def get_child(self, prefix):
        child = None
        for p in prefix:
            child = self.children[p]
        assert child
        return child


def HoeffdingCompatibility(a: FptaNode, b: FptaNode, eps):
    if a.children.keys() != b.children.keys():
        return False

    n1 = sum([child.frequency for child in a.children.values()])
    n2 = sum([child.frequency for child in b.children.values()])

    if n1 > 0 and n2 > 0:
        for o in a.children.keys():
            if abs(a.children[o].frequency / n1 - b.children[o].frequency / n2) > \
                    ((sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / eps))):
                return False
    return True


def create_fpta(data, is_iofpta):
    root_node = FptaNode(data[0][0])
    for seq in data:
        if seq[0] != root_node.output:
            print('All strings should have the same initial output')
            assert False
        curr_node = root_node

        for el in seq[1:]:
            input = el if not is_iofpta else (el[0], el[1])

            if input not in curr_node.children.keys():
                node = FptaNode(el if not is_iofpta else el[1])
                node.prefix = list(curr_node.prefix)
                node.prefix.append(input)
                curr_node.children[input] = node

            curr_node = curr_node.children[input]
            curr_node.frequency += 1

    return root_node, deepcopy(root_node)


class Alergia:
    def __init__(self, data, is_iofpta=False, eps=0.05, round_to=2):
        assert 0 < eps <= 2

        self.t, self.a = create_fpta(data, is_iofpta)

        # self.t, self.a = get_fptas()
        # self.is_iofpta = False

        self.eps = eps
        self.is_iofpta = is_iofpta
        self.round_to = round_to

    def compatibility_test(self, a: FptaNode, b: FptaNode):
        if a.output != b.output:
            return False

        if not a.children.values() or not b.children.values():
            return True

        if not HoeffdingCompatibility(a, b, self.eps):
            return False

        for el in a.children.keys():
            if not self.compatibility_test(a.children[el], b.children[el]):
                return False

        return True

    def merge(self, q_r, q_b):
        t_q_b = self.get_blue_node(q_b)
        prefix_leading_to_state = q_b.prefix[:-1]
        to_update = self.a
        for p in prefix_leading_to_state:
            to_update = to_update.children[p]

        to_update.children[q_b.prefix[-1]] = q_r

        paths = [p.prefix for p in self.get_leaf_nodes(t_q_b)]
        paths = [p[len(t_q_b.prefix):] for p in paths]

        for path in paths:
            state_in_a = q_r
            state_in_t = t_q_b
            for p in path:
                state_in_t = state_in_t.children[p]
                state_in_a = state_in_a.children[p]
                state_in_a.frequency += state_in_t.frequency

    def get_leaf_nodes(self, origin):
        leaves = []

        def _get_leaf_nodes(origin):
            if origin is not None:
                if not origin.children:
                    leaves.append(origin)
                for n in origin.children.values():
                    _get_leaf_nodes(n)

        _get_leaf_nodes(origin)
        return leaves

    def run(self):

        red = {self.a}  # representative nodes and will be included in the final output model
        blue = self.a.succs()  # intermediate successors scheduled for testing

        i = 1
        while blue:
            print(i)
            i += 1
            lex_min_blue = min(list(blue), key=lambda x: len(x.prefix))

            red_sorted = sorted(list(red), key=lambda x: len(x.prefix))

            merged = False

            for q_r in red_sorted:
                if self.compatibility_test(self.get_blue_node(q_r), self.get_blue_node(lex_min_blue)):
                    self.merge(q_r, lex_min_blue)
                    merged = True
                    print("MERGED")
                    break

            if not merged:
                red.add(lex_min_blue)

            blue.clear()
            prefixes_in_red = [s.prefix for s in red]
            for r in red:
                for s in r.succs():
                    if s.prefix not in prefixes_in_red:
                        blue.append(s)

        print(len(red))
        return self.a, red

    def normalize(self, red):
        red_sorted = sorted(list(red), key=lambda x: len(x.prefix))
        for r in red_sorted:
            total_output = sum([c.frequency for c in r.children.values()])
            for i, c in r.children.items():
                if total_output == 0:
                    print('REEEEEE')
                r.children_prob[i] = round(c.frequency / total_output, self.round_to)

    def get_blue_node(self, red_node):
        blue = self.t
        for p in red_node.prefix:
            blue = blue.children[p]
        return blue


def visualize_fpta(red, path="LearnedModel", is_iofpta=False):
    red_sorted = sorted(list(red), key=lambda x: len(x.prefix))
    graph = Dot('fpta', graph_type='digraph')

    for i, r in enumerate(red_sorted):
        r.state_id = f'q{i}'
        graph.add_node(Node(r.state_id, label=r.output))

    for r in red_sorted:
        for i, c in r.children.items():
            graph.add_edge(Edge(r.state_id, c.state_id, label=f'{i[0]}:{r.children_prob[i]}' if is_iofpta else
                                f'{r.children_prob[i]}'))

    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', red_sorted[0].state_id, label=''))

    # graph.write(path=f'{path}.dot', format='raw')
    # exit()
    graph.write(path=f'{path}.pdf', format='pdf')

    try:
        import webbrowser
        abs_path = os.path.abspath(f'{path}.pdf')
        path = f'file:///{abs_path}'
        webbrowser.open(path)
    except:
        print('Err')


def run_Alergia(data, eps=0.05, is_iofpta=False, round_to=2):
    alergia = Alergia(data, eps=eps, is_iofpta=is_iofpta, round_to=round_to)
    root, states = alergia.run()
    alergia.normalize(states)
    return root, states


if __name__ == '__main__':
    #data = [[1, 2, 3, 4], [1, 1, 3, 4], [1, 2, 3, 2]]
    tokenizer = IODelimiterTokenizer()
    data = tokenizer.tokenize_data('mdpData.txt')
    model, states = run_Alergia(data, eps=0.005, is_iofpta=True, round_to=5)

    visualize_fpta(states, is_iofpta=True)

    exit(1)
