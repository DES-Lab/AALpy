import copy
from collections import defaultdict
from queue import Queue
from typing import Set, Dict, Any, List, Tuple
import pydot

from aalpy.automata import StochasticMealyMachine, StochasticMealyState

TransitionID = Tuple[Any, Any]
Prefix = List[TransitionID]

class Node:
    __slots__ = ['transitions', 'prefix', "transition_count"]

    def __init__(self, prefix):
        self.transitions = defaultdict[Any, Dict[Any,Node]](dict)
        self.transition_count = defaultdict[Any, defaultdict[Any, int]](lambda : defaultdict(lambda : 0))
        self.prefix : Prefix = prefix

    def __lt__(self, other):
        return len(self.prefix) < len(other.prefix)

    def __le__(self, other):
        return len(self.prefix) <= len(other.prefix)

    def __eq__(self, other):
        return self.prefix == other.prefix

    def __hash__(self):
        return id(self)  # TODO This is a hack

    def count(self):
        return sum((sum(items.values()) for items in self.transition_count.values()))

    def transition_iterator(self):
        for in_sym, transitions in self.transitions.items():
            for out_sym, node in transitions.items():
                yield (in_sym, out_sym), node

    def shallow_copy(self) -> 'Node':
        node = Node(self.prefix)
        for in_sym, t in self.transitions.items():
            node.transitions[in_sym] = dict(t)
        node.transition_count = copy.deepcopy(self.transition_count)
        return node

    def get_by_prefix(self, seq : Prefix) -> 'Node':
        node = self
        for in_sym, out_sym in seq:
            if in_sym is None:
                continue
            node = node.transitions[in_sym][out_sym]
        return node

    def get_all_nodes(self) -> Set['Node']:
        qu = Queue[Node]()
        qu.put(self)
        nodes = set()
        while not qu.empty():
            state = qu.get()
            nodes.add(state)
            for _, child in state.transition_iterator():
                if child not in nodes:
                    qu.put(child)
        return nodes

    def to_automaton(self) -> StochasticMealyMachine:
        #TODO fix
        nodes = self.get_all_nodes()
        nodes.remove(self)  # dunno whether order is preserved?
        nodes = [self] + list(nodes)

        state_map = dict()
        for i, r in enumerate(nodes):
            state = StochasticMealyState(f's{i}')
            state_map[r] = state
            state.prefix = r.prefix

        initial_state = state_map[self]

        for r in nodes:
            for in_sym, transitions in r.transitions.items():
                count = r.transition_count[in_sym]
                total = sum(count.values())
                for out_sym, new_state in transitions.items():
                    transition = (state_map[new_state], out_sym, count[out_sym] / total)
                    state_map[r].transitions[in_sym].append(transition)

        return StochasticMealyMachine(initial_state, list(state_map.values()))

    def visualize(self, path : str):
        graph = pydot.Dot('fpta', graph_type='digraph')

        graph.add_node(pydot.Node(str(self.prefix), label=f'{self.count()}'))

        nodes = self.get_all_nodes()

        for node in nodes:
            graph.add_node(pydot.Node(str(node.prefix), label=f'{node.count()}'))

        for node in nodes:
            for in_sym, options in node.transitions.items():
                if 1 < len(options) :
                    color = 'red'
                else:
                    color = 'black'

                for out_sym, c in options.items():
                    graph.add_edge(pydot.Edge(str(node.prefix), str(c.prefix), label=f'{in_sym} / {out_sym} [{node.transition_count[in_sym][out_sym]}]', color=color, fontcolor=color))

        graph.add_node(pydot.Node('__start0', shape='none', label=''))
        graph.add_edge(pydot.Edge('__start0', str(self.prefix), label=''))

        graph.write(path=path, format='pdf')

    def add_data(self, data):
        for seq in data:
            curr_node : Node = self
            for in_sym, out_sym in seq:
                curr_node.transition_count[in_sym][out_sym] += 1
                transitions = curr_node.transitions[in_sym]
                if out_sym not in transitions:
                    node = Node(curr_node.prefix + [(in_sym, out_sym)])
                    transitions[out_sym] = node
                else:
                    node = transitions[out_sym]
                curr_node = node
    @staticmethod
    def createPTA(data, initial_output = None) :
        root_node = Node([(None, initial_output)])
        root_node.add_data(data)
        return root_node

    def is_locally_deterministic(self):
        return all(len(item) == 1 for item in self.transitions.values())

    def is_deterministic(self):
        return all(node.is_locally_deterministic() for node in self.get_all_nodes())

    def deterministic_compatible(self, other : 'Node'):
        common_keys = (key for key in self.transitions.keys() if key in other.transitions.keys())
        return all(list(self.transitions[key].keys()) == list(other.transitions[key].keys()) for key in common_keys)

    def is_moore(self):
        output_dict = dict()
        for node in self.get_all_nodes():
            for (in_sym, out_sym), child in node.transition_iterator():
                if child in output_dict.keys() and output_dict[child] != out_sym:
                    return False
                else:
                    output_dict[child] = out_sym
        return True

    def moore_compatible(self, other : 'Node'):
        return self.prefix[-1][1] == other.prefix[-1][1]
