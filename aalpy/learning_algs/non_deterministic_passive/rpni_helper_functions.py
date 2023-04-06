from collections import defaultdict
from queue import Queue
from typing import Set, Dict, Any, List, Tuple
import pydot


class Node:
    __slots__ = ['output', 'transitions', 'prefix', "type"]

    def __init__(self, output, prefix):
        self.output = output
        self.transitions : Dict[Any,Dict[Any,Node]] = defaultdict(lambda : dict())
        self.prefix = prefix

    def __lt__(self, other):
        return len(self.prefix) < len(other.prefix)

    def __le__(self, other):
        return len(self.prefix) <= len(other.prefix)

    def __eq__(self, other):
        return self.prefix == other.prefix

    def __hash__(self):
        return id(self)  # TODO This is a hack

    def shallow_copy(self):
        node = Node(self.output, self.prefix)
        for in_sym, options in self.transitions.items():
            node.transitions[in_sym] = dict(options)
        return node

    def get_by_prefix(self, seq : List[Tuple[Any, Any]]) -> 'Node':
        node = self
        for in_sym, out_sym in seq:
            node = node.transitions[in_sym][out_sym]
        return node

    def get_children(self):
        return [node for options in self.transitions.values() for node in options.values()]

    def get_all_nodes(self) -> Set['Node']:
        qu = Queue()
        qu.put(self)
        nodes = set()
        while not qu.empty():
            state = qu.get()
            nodes.add(state)
            for child in state.get_children():
                if child not in nodes:
                    qu.put(child)
        return nodes

    def to_automaton(self):
        nodes = self.get_all_nodes()
        nodes.remove(self)  # dunno whether order is preserved?
        nodes = [self] + list(nodes)

        from aalpy.automata.NonDeterministicMooreMachine import NDMooreMachine, NDMooreState

        state_map = dict()
        for i, r in enumerate(nodes):
            state_map[r] = NDMooreState(f's{i}', r.output)

        initial_state = state_map[nodes[0]]

        for r in nodes:
            for in_sym, options in r.transitions.items():
                for out_sym, c in options.items():
                    state_map[r].transitions[in_sym].append(state_map[c])

        return NDMooreMachine(initial_state, list(state_map.values()))

    def compatible_outputs(self, other : 'Node'):
        return self.output == other.output

    def nondeterministic_additions(self, other : 'Node'):
        count = 0
        for in_sym, opts in self.transitions.items():
            if in_sym not in other.transitions.keys():
                continue
            count += sum(out_sym not in opts.keys() for out_sym in other.transitions[in_sym])
        return count

    def visualize(self, path : str):
        graph = pydot.Dot('fpta', graph_type='digraph')

        graph.add_node(pydot.Node(str(self.prefix), label=f'{self.output}'))

        queue : List[Node] = [self]
        visited = set()
        visited.add(self)
        while queue:
            curr = queue.pop(0)
            for in_sym, options in curr.transitions.items():
                if 1 < len(options) :
                    color = 'red'
                else:
                    color = 'black'

                for out_sym, c in options.items():
                    if c not in visited:
                        graph.add_node(pydot.Node(str(c.prefix), label=f'{c.output}'))
                        queue.append(c)
                        visited.add(c)
                    graph.add_edge(pydot.Edge(str(curr.prefix), str(c.prefix), label=f'{in_sym}', color=color))

        graph.add_node(pydot.Node('__start0', shape='none', label=''))
        graph.add_edge(pydot.Edge('__start0', str(self.prefix), label=''))

        graph.write(path=path, format='pdf')

def createPTA(data) :

    root_node = Node(data[0][0], tuple())
    for seq in data:
        if not seq[0] == root_node.output:
            raise ValueError("conflicting initial outputs")

        curr_node = root_node
        for in_sym, out_sym in seq[1:]:
            options = curr_node.transitions[in_sym]
            if out_sym not in options.keys():
                node = Node(out_sym, curr_node.prefix + ((in_sym,out_sym),))
                options[out_sym] = node

            curr_node = options[out_sym]

    return root_node
