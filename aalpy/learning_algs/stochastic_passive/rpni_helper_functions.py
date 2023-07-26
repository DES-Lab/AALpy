import copy
import pathlib
from collections import defaultdict
from functools import total_ordering
from queue import Queue
from typing import Set, Dict, Any, List, Tuple, Literal, Iterable, Callable, NamedTuple
import pydot

from aalpy.automata import StochasticMealyMachine, StochasticMealyState, MooreState, MooreMachine, NDMooreState, \
    NDMooreMachine, Mdp, MdpState, MealyMachine, MealyState, Onfsm, OnfsmState
from aalpy.base import Automaton

OutputBehavior = Literal["moore", "mealy"]
TransitionBehavior = Literal["deterministic", "nondeterministic", "stochastic"]

IOPair = NamedTuple("IOPair", [("input", Any), ("output",Any)])
Prefix = List[IOPair]

IOTrace = Iterable[IOPair]
IOExample = Tuple[Iterable[Any], Any]

StateFunction = Callable[['Node'], str]
TransitionFunction = Callable[['Node', Any, Any], str]

# Separate functions to allow pickling Nodes
def zero():
    return 0
def create_count_dict():
    return defaultdict(zero)

@total_ordering
class Node:
    """
    Generic class for observably deterministic automata.

    The prefix is given as (minimal) list of IO pairs leading to that state.
    We assume an initial transition to the initial state, which has to be reflected in the prefix.
    This way, the output of the initial state for Moore machines can be encoded in its prefix.

    Transition count is preferred over state count as it allows to easily count transitions for non-tree-shaped automata
    """
    __slots__ = ['transitions', 'prefix', "transition_count"]

    def __init__(self, prefix : Prefix):
        self.transitions = defaultdict[Any, Dict[Any,Node]](dict)
        self.transition_count = defaultdict[Any, defaultdict[Any, int]](create_count_dict)
        self.prefix : Prefix = prefix

    def __lt__(self, other):
        # TODO maybe check whether inputs / outputs implement __lt__
        if len(self.prefix) != len(other.prefix):
            return len(self.prefix) < len(other.prefix)

        class SafeCmp:
            def __init__(self, val):
                self.val = val

            def __lt__(self, other):
                try:
                    return self.val < other.val
                except TypeError:
                    return str(self.val) < str(other.val)

        s_prefix, o_prefix = ([(SafeCmp(i), SafeCmp(o)) for i,o in x.prefix] for x in [self, other])
        return s_prefix < o_prefix

    def __eq__(self, other):
        return self.prefix == other.prefix

    def __hash__(self):
        return id(self) # TODO This is a hack

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

    def to_automaton(self, output_behavior : OutputBehavior, transition_behavior : TransitionBehavior, check_behavior = False) -> Automaton:
        nodes = self.get_all_nodes()

        if check_behavior:
            if output_behavior == "moore" and not self.is_moore():
                raise ValueError("Tried to obtain Moore machine from non-Moore structure")
            if transition_behavior == "deterministic" and not self.is_deterministic():
                raise ValueError("Tried to obtain deterministic automaton from non-deterministic structure")

        type_dict = {
            ("moore","deterministic") : (MooreMachine, MooreState),
            ("moore","nondeterministic") : (NDMooreMachine, NDMooreState),
            ("moore","stochastic") : (Mdp, MdpState),
            ("mealy","deterministic") : (MealyMachine, MealyState),
            ("mealy","nondeterministic") : (Onfsm, OnfsmState),
            ("mealy","stochastic") : (StochasticMealyMachine, StochasticMealyState),
        }

        Machine, State = type_dict[(output_behavior, transition_behavior)]

        # create states
        state_map = dict()
        for i, node in enumerate(nodes):
            state_id = f's{i}'
            match output_behavior:
                case "mealy" : state = State(state_id)
                case "moore" : state = State(state_id, node.prefix[-1].output)
            state_map[node] = state
            match transition_behavior:
                case "deterministic": state.prefix = tuple(p.input for p in node.prefix)
                case _: state.prefix = tuple(node.prefix)

        initial_state = state_map[self]

        # add transitions
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

    def visualize(self, path : str | pathlib.Path, output_behavior : OutputBehavior = "mealy", produce_pdf : bool = False, engine = "dot", *,
                  state_label : StateFunction = None, state_color : StateFunction = None,
                  trans_label : TransitionFunction = None, trans_color : TransitionFunction = None,
                  state_props : dict[str,StateFunction] = None,
                  trans_props : dict[str,TransitionFunction] = None):

        # handle default parameters
        if output_behavior not in ["moore", "mealy", None]:
            raise ValueError(f"Invalid OutputBehavior {output_behavior}")
        if state_props is None:
            state_props = dict()
        if trans_props is None:
            trans_props = dict()
        if state_label is None:
            match output_behavior:
                case "moore": state_label = lambda node : f'{node.prefix[-1][1]} {node.count()}'
                case _: state_label = lambda node: f'{node.count()}'
        if trans_label is None and "label" not in trans_props:
            match output_behavior:
                case "moore": trans_label = lambda node, in_sym, out_sym : f'{in_sym} [{node.transition_count[in_sym][out_sym]}]'
                case _: trans_label = lambda node, in_sym, out_sym : f'{in_sym} / {out_sym} [{node.transition_count[in_sym][out_sym]}]'
        if state_color is None:
            state_color = lambda x : "black"
        if trans_color is None:
            trans_color = lambda x, y, z : "black"
        state_props = {"label" : state_label, "color" : state_color , "fontcolor" : state_color, **state_props}
        trans_props = {"label" : trans_label, "color" : trans_color, "fontcolor" : trans_color, **trans_props}

        # create new graph
        graph = pydot.Dot('fpta', graph_type='digraph')

        #graph.add_node(pydot.Node(str(self.prefix), label=state_label(self)))
        nodes = self.get_all_nodes()

        # add nodes
        for node in nodes:
            arg_dict = {key : fun(node) for key, fun in state_props.items()}
            graph.add_node(pydot.Node(str(node.prefix), **arg_dict))

        # add transitions
        for node in nodes:
            for in_sym, options in node.transitions.items():
                for out_sym, c in options.items():
                    arg_dict = {key : fun(node, in_sym, out_sym) for key, fun in trans_props.items()}
                    graph.add_edge(pydot.Edge(str(node.prefix), str(c.prefix), **arg_dict))

        # add initial state
        # TODO maybe add option to parameterize this
        graph.add_node(pydot.Node('__start0', shape='none', label=''))
        graph.add_edge(pydot.Edge('__start0', str(self.prefix), label=''))

        format = 'pdf' if produce_pdf else 'raw'
        file_ext = 'pdf' if produce_pdf else 'dot'
        graph.write(path=str(path) + "." + file_ext, prog=engine, format=format)

    def add_data(self, data):
        for seq in data:
            self.add_trace(seq)

    def add_trace(self, data : IOTrace):
        curr_node : Node = self
        for in_sym, out_sym in data:
            curr_node.transition_count[in_sym][out_sym] += 1
            transitions = curr_node.transitions[in_sym]
            if out_sym not in transitions:
                node = Node(curr_node.prefix + [IOPair(in_sym, out_sym)])
                transitions[out_sym] = node
            else:
                node = transitions[out_sym]
            curr_node = node

    def add_example(self, data : IOExample):
        # TODO add support for example based algorithms
        raise NotImplementedError("This is a work in progress")
        seq, output = data
        curr_node : Node = self
        for sym in seq:
            curr_node.transition_count[sym][None] += 1
            t = curr_node.transitions[sym]
            if None not in t:
                t[None] = Node(curr_node.prefix + [IOPair(sym, None)])
            curr_node = t[None]
        curr_node.prefix[-1] = IOPair(seq[-1] if len(seq) != 0 else None, output)

    @staticmethod
    def createPTA(data, initial_output = None) :
        root_node = Node([IOPair(None, initial_output)])
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
