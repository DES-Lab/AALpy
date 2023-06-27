import copy
from collections import defaultdict
from functools import total_ordering
from queue import Queue
from typing import Set, Dict, Any, List, Tuple, Literal
import pydot

from aalpy.automata import StochasticMealyMachine, StochasticMealyState, MooreState, MooreMachine, NDMooreState, \
    NDMooreMachine, Mdp, MdpState, MealyMachine, MealyState, Onfsm, OnfsmState
from aalpy.base import Automaton

TransitionID = Tuple[Any, Any]
Prefix = List[TransitionID]

OutputBehavior = Literal["moore", "mealy"]
TransitionBehavior = Literal["deterministic", "nondeterministic", "stochastic"]

def zero():
    return 0
def create_count_dict():
    return defaultdict(zero)

@total_ordering
class Node:
    __slots__ = ['transitions', 'prefix', "transition_count"]

    def __init__(self, prefix):
        self.transitions = defaultdict[Any, Dict[Any,Node]](dict)
        self.transition_count = defaultdict[Any, defaultdict[Any, int]](create_count_dict)
        self.prefix : Prefix = prefix

    def __lt__(self, other):
        # TODO maybe check whether inputs / outputs implement __lt__
        s_prefix, o_prefix = ([(str(i), str(o)) for i,o in x.prefix[1:]] for x in [self, other])
        return (len(s_prefix), s_prefix) < (len(o_prefix), o_prefix)

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
        nodes = {self}
        while not qu.empty():
            state = qu.get()
            for _, child in state.transition_iterator():
                if child not in nodes:
                    qu.put(child)
                    nodes.add(child)
        return nodes

    def to_automaton(self, output_behavior : OutputBehavior, transition_behavior : TransitionBehavior) -> Automaton:
        nodes = self.get_all_nodes()

        type_dict = {
            ("moore","deterministic") : (MooreMachine, MooreState),
            ("moore","nondeterministic") : (NDMooreMachine, NDMooreState),
            ("moore","stochastic") : (Mdp, MdpState),
            ("mealy","deterministic") : (MealyMachine, MealyState),
            ("mealy","nondeterministic") : (Onfsm, OnfsmState),
            ("mealy","stochastic") : (StochasticMealyMachine, StochasticMealyState),
        }

        Machine, State = type_dict[(output_behavior, transition_behavior)]

        state_map = dict()
        for i, node in enumerate(nodes):
            state_id = f's{i}'
            match output_behavior:
                case "mealy" : state = State(state_id)
                case "moore" : state = State(state_id, node.prefix[-1][1])
            state_map[node] = state
            state.prefix = node.prefix

        initial_state = state_map[self]

        for node in nodes:
            state = state_map[node]
            for in_sym, transitions in node.transitions.items():
                count = node.transition_count[in_sym]
                total = sum(count.values())
                for out_sym, target_node in transitions.items():
                    target_state = state_map[target_node]
                    match (output_behavior, transition_behavior):
                        case ("moore","deterministic") :
                            state.transitions[in_sym] = target_state
                        case ("mealy","deterministic") :
                            state.transitions[in_sym] = target_state
                            state.output_fun[in_sym] = out_sym
                        case ("moore","nondeterministic"):
                            state.transitions[in_sym].append(target_state)
                        case ("mealy","nondeterministic"):
                            state.transitions[in_sym].append((out_sym, target_state))
                        case ("moore","stochastic"):
                            state.transitions[in_sym].append((target_state, count[out_sym] / total))
                        case ("mealy","stochastic"):
                            state.transitions[in_sym].append((target_state, out_sym, count[out_sym] / total))

        return Machine(initial_state, list(state_map.values()))

    def visualize(self, path : str, output_behavior : OutputBehavior, produce_pdf : bool = False):
        graph = pydot.Dot('fpta', graph_type='digraph')

        match output_behavior:
            case "moore":
                state_label = lambda node : f'{node.prefix[-1][1]} {node.count()}'
                transition_label = lambda node, in_sym, out_sym : f'{in_sym} [{node.transition_count[in_sym][out_sym]}]'
            case "mealy":
                state_label = lambda node: f'{node.count()}'
                transition_label = lambda node, in_sym, out_sym : f'{in_sym} / {out_sym} [{node.transition_count[in_sym][out_sym]}]'

        graph.add_node(pydot.Node(str(self.prefix), label=state_label(self)))

        nodes = self.get_all_nodes()

        for node in nodes:
            graph.add_node(pydot.Node(str(node.prefix), label=state_label(node)))

        for node in nodes:
            for in_sym, options in node.transitions.items():
                if 1 < len(options) :
                    color = 'red'
                else:
                    color = 'black'

                for out_sym, c in options.items():
                    graph.add_edge(pydot.Edge(str(node.prefix), str(c.prefix), label=transition_label(node,in_sym,out_sym), color=color, fontcolor=color))

        graph.add_node(pydot.Node('__start0', shape='none', label=''))
        graph.add_edge(pydot.Edge('__start0', str(self.prefix), label=''))

        format = 'pdf' if produce_pdf else 'raw'
        graph.write(path=path, format=format)

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
