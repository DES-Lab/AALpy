import pickle
import queue
from functools import total_ordering
from typing import Set

from aalpy.automata import VpaState, VpaTransition, Vpa


@total_ordering
class VpaRpniNode:
    __slots__ = ['output', 'children', 'prefix', 'type', 'top_of_stack']

    def __init__(self, output=None, children=None):
        if children is None:
            children = dict()
        self.output = output
        self.children = children
        self.prefix = ()
        self.top_of_stack = dict()

    def copy(self):
        return pickle.loads(pickle.dumps(self, -1))

    def __lt__(self, other):
        return (len(self.prefix), self.prefix) < (len(other.prefix), other.prefix)

    def __eq__(self, other):
        return self.prefix == other.prefix

    def __hash__(self):
        return id(self)  # TODO This is a hack

    def get_all_nodes(self) -> Set['VpaRpniNode']:
        qu = queue.Queue()
        qu.put(self)
        nodes = set()
        while not qu.empty():
            state = qu.get()
            nodes.add(state)
            for child in state.children.values():
                if child not in nodes:
                    qu.put(child)
        return nodes

    def to_automaton(self):
        nodes = self.get_all_nodes()
        nodes.remove(self)  # dunno whether order is preserved?
        nodes = [self] + list(nodes)
        return to_vpa(nodes, self.type)

    def compatible_outputs(self, other):
        so, oo = [self.output, other.output]
        cmp = lambda x, y: x is None or y is None or x == y
        if self.type == 'moore':
            return cmp(so, oo)
        else:
            return all(cmp(so[key], oo[key]) for key in filter(lambda k: k in oo, so))

    def get_child_by_prefix(self, prefix):
        node = self
        for symbol in prefix:
            node = node.children[symbol]
        return node


def create_Vpa_PTA(data, vpa_alphabet):
    data.sort(key=lambda x: len(x[0]))

    root_node = VpaRpniNode()
    for seq, label in data:
        curr_node = root_node
        stack = []
        for idx, symbol in enumerate(seq):
            if symbol not in curr_node.children.keys():
                node = VpaRpniNode()
                node.prefix = curr_node.prefix + (symbol,)
                curr_node.children[symbol] = node

            curr_node = curr_node.children[symbol]

            if symbol in vpa_alphabet.call_alphabet:
                stack.append(symbol)
            elif symbol in vpa_alphabet.return_alphabet:
                top_of_stack = stack.pop()
                curr_node.top_of_stack = {symbol: top_of_stack}

        if curr_node.output is None:
            curr_node.output = label
        if curr_node.output != label:
            return None

    #visualize_vpa_pta(root_node)
    #exit()
    return root_node


def extract_unique_sequences(root_node):
    def get_leaf_nodes(root):
        leaves = []

        def _get_leaf_nodes(node):
            if node is not None:
                if len(node.children.keys()) == 0:
                    leaves.append(node)
                for n in node.children.values():
                    _get_leaf_nodes(n)

        _get_leaf_nodes(root)
        return leaves

    leaf_nodes = get_leaf_nodes(root_node)
    paths = []
    for node in leaf_nodes:
        seq = []
        curr_node = root_node
        for i in node.prefix:
            curr_node = curr_node.children[i]
            seq.append((i, curr_node.output))
        paths.append(seq)

    return paths


def to_vpa(red, vpa_alphabet=None):
    initial_state = None
    prefix_state_map = {}
    for i, r in enumerate(red):
        prefix_state_map[r.prefix] = VpaState(f's{i}', r.output)
        if i == 0:
            initial_state = prefix_state_map[r.prefix]

    for r in red:
        for i, c in r.children.items():
            action = 'push' if i in vpa_alphabet.call_alphabet else 'pop' if i in vpa_alphabet.return_alphabet else None
            stack_guard = i if action == 'push' else c.top_of_stack[i] if action == 'pop' else None
            transition = VpaTransition(prefix_state_map[r.prefix], prefix_state_map[c.prefix], i, action, stack_guard)
            prefix_state_map[r.prefix].transitions[i].append(transition)
    return Vpa(initial_state, list(prefix_state_map.values()), vpa_alphabet)


def check_vpa_sequence(root_node, seq, vpa_alphabet):
    """
    Checks whether each sequence in the dataset is valid in the current automaton.
    """
    curr_node = root_node
    stack = []
    for i, o in seq:
        # check if outputs are the same, iff output in test data is concrete (not None)
        curr_node = curr_node.children[i]
        if o is not None and curr_node.output != o:
            return False

        if i in vpa_alphabet.call_alphabet:
            stack.append(i)

        elif i in vpa_alphabet.return_alphabet:
            top_of_stack = stack.pop()
            if top_of_stack != curr_node.top_of_stack[i]:
                return False

    return stack == []


def visualize_vpa_pta(root_node, path='pta.pdf'):
    from pydot import Dot, Node, Edge
    graph = Dot('fpta', graph_type='digraph')

    graph.add_node(Node(str(root_node.prefix), label=f'{root_node.output}'))

    queue = [root_node]
    visited = set()
    visited.add(root_node.prefix)
    while queue:
        curr = queue.pop(0)
        for i, c in curr.children.items():
            if c.prefix not in visited:
                output = f'{c.output}'
                if c.top_of_stack:
                    output += f'\nTop Of Stack: {c.top_of_stack}'
                graph.add_node(Node(str(c.prefix), label=output))
            graph.add_edge(Edge(str(curr.prefix), str(c.prefix), label=f'{i}'))
            if c.prefix not in visited:
                queue.append(c)
            visited.add(c.prefix)

    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', str(root_node.prefix), label=''))

    graph.write(path=path, format='pdf')
