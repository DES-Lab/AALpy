from collections import defaultdict
from queue import Queue
from typing import Set, Dict, Any, List, Tuple
import pydot
from aalpy.automata.NonDeterministicMooreMachine import NDMooreMachine, NDMooreState


class Node:
    __slots__ = ['output', 'count', 'transitions', 'prefix', "type"]

    def __init__(self, output, prefix):
        self.count = 0
        self.output = output
        self.transitions : Dict[Tuple[Any,Any], Node] = dict()
        self.prefix : List[Tuple[Any,Any]] = prefix

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
        node.transitions = dict(self.transitions)
        return node

    def get_by_prefix(self, seq : List[Tuple[Any, Any]]) -> 'Node':
        node = self
        for sym_pair in seq:
            node = node.transitions[sym_pair]
        return node

    def get_all_nodes(self) -> Set['Node']:
        qu = Queue[Node]()
        qu.put(self)
        nodes = set()
        while not qu.empty():
            state = qu.get()
            nodes.add(state)
            for child in state.transitions.values():
                if child not in nodes:
                    qu.put(child)
        return nodes

    def to_automaton(self):
        nodes = self.get_all_nodes()
        nodes.remove(self)  # dunno whether order is preserved?
        nodes = [self] + list(nodes)

        state_map = dict()
        for i, r in enumerate(nodes):
            state_map[r] = NDMooreState(f's{i}', r.output)

        initial_state = state_map[nodes[0]]

        for r in nodes:
            for (in_sym, out_sym), c in r.transitions.items():
                state_map[r].transitions[in_sym].append(state_map[c])

        return NDMooreMachine(initial_state, list(state_map.values()))

    def compatible_outputs(self, other : 'Node'):
        return self.output == other.output

    def get_two_stage_transition_dict(self):
        transitions = defaultdict(lambda : dict())
        for (in_sym, out_sym), child in self.transitions.items():
            transitions[in_sym][out_sym] = child
        return transitions

    def nondeterministic_additions(self, other : 'Node'):
        count = 0
        st = self.get_two_stage_transition_dict()
        ot = other.get_two_stage_transition_dict()

        for in_sym, opts in st.items():
            if in_sym not in ot:
                continue
            count += sum(out_sym not in opts.keys() for out_sym in ot[in_sym])
        return count

    def visualize(self, path : str, data = None, pta = None):
        graph = pydot.Dot('fpta', graph_type='digraph')

        transition_count = defaultdict(lambda : 0)
        if data is None:
            data = []
        elif pta is None:
            pta = Node.createPTA(data)
        for seq in data:
            current = self
            current_pta = pta
            for sym_pair in seq[1:]:
                if sym_pair not in current_pta.transitions:
                    break
                if sym_pair not in current.transitions:
                    raise AssertionError("no lang inclusion")
                transition_count[(current,sym_pair)] += 1
                current = current.transitions[sym_pair]
                current_pta = current_pta.transitions[sym_pair]
                if current.output != sym_pair[1]:
                    raise AssertionError("wrong output")

        graph.add_node(pydot.Node(str(self.prefix), label=f'{self.output}'))

        queue : List[Node] = [self]
        visited = set()
        visited.add(self)
        while queue:
            curr = queue.pop(0)
            for in_sym, options in curr.get_two_stage_transition_dict().items():
                if 1 < len(options) :
                    color = 'red'
                else:
                    color = 'black'

                for out_sym, c in options.items():
                    if c not in visited:
                        graph.add_node(pydot.Node(str(c.prefix), label=f'{c.output}'))
                        queue.append(c)
                        visited.add(c)
                    graph.add_edge(pydot.Edge(str(curr.prefix), str(c.prefix), label=f'{in_sym} [{transition_count[(curr,(in_sym,out_sym))]}]', color=color, fontcolor=color))

        graph.add_node(pydot.Node('__start0', shape='none', label=''))
        graph.add_edge(pydot.Edge('__start0', str(self.prefix), label=''))

        graph.write(path=path, format='pdf')

    def prune(self, threshold = 20):
        q = Queue[Node]()
        q.put(self)
        while not q.empty():
            node = q.get()
            if node.count < threshold:
                node.transitions.clear()

            for child in node.transitions.values():
                q.put(child)

    @staticmethod
    def createPTA(data) :

        root_node = Node(data[0][0], [])
        root_node.count = len(data)
        for seq in data:
            if not seq[0] == root_node.output:
                raise ValueError("conflicting initial outputs")

            curr_node = root_node
            for in_sym, out_sym in seq[1:]:
                sym_pair = (in_sym, out_sym)
                if sym_pair not in curr_node.transitions:
                    node = Node(out_sym, curr_node.prefix + [sym_pair])
                    curr_node.transitions[sym_pair] = node
                else:
                    node = curr_node.transitions[sym_pair]
                node.count += 1
                curr_node = node

        return root_node
