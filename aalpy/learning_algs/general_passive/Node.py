import itertools
import math
import pathlib
from collections import deque
from enum import Enum
from functools import total_ordering
from typing import Dict, Any, List, Tuple, Iterable, Callable, Union, Set, TypeVar, Iterator
import pydot
from copy import copy

from aalpy.automata import StochasticMealyMachine, StochasticMealyState, MooreState, MooreMachine, NDMooreState, \
    NDMooreMachine, Mdp, MdpState, MealyMachine, MealyState, Onfsm, OnfsmState
from aalpy.base import Automaton

Key = TypeVar("Key")
Val = TypeVar("Val")

OutputBehavior = str
OutputBehaviorRange = ["moore", "mealy"]

TransitionBehavior = str
TransitionBehaviorRange = ["deterministic", "nondeterministic", "stochastic"]

IOPair = Tuple[Any, Any]
IOTrace = List[IOPair]
IOExample = Tuple[Iterable[Any], Any]

StateFunction = Callable[['Node'], str]
TransitionFunction = Callable[['Node', Any, Any], str]

def generate_values(base : list, step : Callable, backing_set=True):
    if backing_set:
        result = list(base)
        control = set(base)
        for val in result:
            for new_val in step(val):
                if new_val not in control:
                    control.add(new_val)
                    result.append(new_val)
        return result
    else:
        result = list(base)
        for val in result:
            for new_val in step(val):
                if new_val not in result:
                    result.append(new_val)
        return result

def intersection_iterator(a: Dict[Key, Val], b: Dict[Key, Val]) -> Iterator[Tuple[Key, Val, Val]]:
    missing = object()
    for key, a_val in a.items():
        b_val = b.get(key, missing)
        if b_val is missing:
            continue
        yield key, a_val, b_val

def join_iterator(a: Dict[Key, Val], b: Dict[Key, Val], default: Val = None) -> Iterator[Tuple[Key, Val, Val]]:
    for key, a_val in a.items():
        b_val = b.get(key, default)
        yield key, a_val, b_val
    for key, b_val in b.items():
        if key in a:
            continue
        a_val = a.get(key, default)
        yield key, a_val, b_val

# TODO maybe split this for maintainability (and perfomance?)
class TransitionInfo:
    __slots__ = ["target", "count", "original_target", "original_count"]

    def __init__(self, target, count, original_target, original_count):
        self.target : 'Node' = target
        self.count : int = count
        self.original_target : 'Node' = original_target
        self.original_count : int = original_count

# TODO add custom pickling code that flattens the Node structure in order to circumvent running into recursion issues for large models
@total_ordering
class Node:
    """
    Generic class for observably deterministic automata.

    The prefix is given as (minimal) list of IO pairs leading to that state.
    We assume an initial transition to the initial state, which has to be reflected in the prefix.
    This way, the output of the initial state for Moore machines can be encoded in its prefix.

    Transition count is preferred over state count as it allows to easily count transitions for non-tree-shaped automata
    """
    __slots__ = ['transitions', 'predecessor', 'prefix_access_pair']

    def __init__(self, prefix_access_pair, predecessor : 'Node' = None):
        # TODO try single dict
        self.transitions : Dict[Any, Dict[Any, TransitionInfo]] = dict()
        self.predecessor : Node = predecessor
        self.prefix_access_pair = prefix_access_pair

    def __lt__(self, other, compare_length_only=False):
        own_l, other_l = self.get_prefix_length(), other.get_prefix_length()
        if own_l != other_l:
            return own_l < other_l
        if compare_length_only:
            return False
        own_p = self.get_prefix()
        other_p = other.get_prefix()
        try:
            return own_p < other_p
        except TypeError:
            return [str(x) for x in own_p] < [str(x) for x in other_p]

    def __eq__(self, other):
        return self is other # TODO hack, does this lead to problems down the line?

    def __hash__(self):
        return id(self) # TODO This is a hack

    # TODO implicit prefixes as currently implemented require O(length) time for prefix calculations (e.g. to determine the minimal blue node)
    # other options would be to have more efficient explicit prefixes such as shared list representations
    def get_prefix_length(self):
        node = self
        length = 0
        while node.predecessor:
            node = node.predecessor
            length += 1
        return length

    def get_prefix_output(self):
        return self.prefix_access_pair[1]

    def get_prefix(self, include_output=True):
        node = self
        prefix = []
        while node.predecessor:
            symbol = node.prefix_access_pair
            if not include_output:
                symbol = symbol[0]
            prefix.append(symbol)
            node = node.predecessor
        prefix.reverse()
        return prefix

    def get_or_create_transitions(self, in_sym) -> Dict[Any, TransitionInfo]:
        t = self.transitions.get(in_sym)
        if t is None:
            t = dict()
            self.transitions[in_sym] = t
        return t

    def transition_iterator(self) -> Iterable[Tuple[Tuple[Any, Any], TransitionInfo]]:
        for in_sym, transitions in self.transitions.items():
            for out_sym, node in transitions.items():
                yield (in_sym, out_sym), node

    def shallow_copy(self) -> 'Node':
        node = Node(self.prefix_access_pair, self.predecessor)
        for in_sym, t in self.transitions.items():
            d = dict()
            for out_sym, v in t.items():
                d[out_sym] = copy(v)
            node.transitions[in_sym] = d
        return node

    def get_by_prefix(self, seq : IOTrace) -> 'Node':
        node : Node = self
        for in_sym, out_sym in seq:
            if in_sym is None: # ignore initial transition of Node.get_prefix()
                continue
            trans = node.transitions.get(in_sym)
            if trans is None:
                return None
            t_info = trans.get(out_sym)
            if t_info is None:
                return None
            node = t_info.target
        return node

    def get_all_nodes(self) -> List['Node']:
        def generator(state : Node):
            for _, child in state.transition_iterator():
                yield child.target
        return generate_values([self], generator)

    def is_tree(self):
        q : List['Node'] = [self]
        backing_set = set()
        while len(q) != 0:
            current = q.pop(0)
            for _, child in current.transition_iterator():
                t = child.target
                if t in backing_set:
                    return False
                q.append(t)
                backing_set.add(t)
        return True

    def to_automaton(self, output_behavior : OutputBehavior, transition_behavior : TransitionBehavior, check_behavior = False, set_prefix = False) -> Automaton:
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

        AutomatonClass, StateClass = type_dict[(output_behavior, transition_behavior)]

        # create states
        state_map = dict()
        for i, node in enumerate(nodes):
            state_id = f's{i}'
            if output_behavior == "mealy":
                state = StateClass(state_id)
            elif output_behavior == "moore":
                state = StateClass(state_id, node.get_prefix_output())
            state_map[node] = state
            if set_prefix:
                if transition_behavior == "deterministic":
                    state.prefix = tuple(p[0] for p in node.get_prefix())
                else:
                    state.prefix = tuple(node.get_prefix())
            else:
                state.prefix = None

        initial_state = state_map[self]

        # add transitions
        for node in nodes:
            state = state_map[node]
            for in_sym, transitions in node.transitions.items():
                total = sum(t.count for t in transitions.values())
                for out_sym, target_node in transitions.items():
                    target_state = state_map[target_node.target]
                    count = target_node.count
                    if AutomatonClass is MooreMachine:
                        state.transitions[in_sym] = target_state
                    elif AutomatonClass is MealyMachine :
                        state.transitions[in_sym] = target_state
                        state.output_fun[in_sym] = out_sym
                    elif AutomatonClass is NDMooreMachine:
                        state.transitions[in_sym].append(target_state)
                    elif AutomatonClass is Onfsm:
                        state.transitions[in_sym].append((out_sym, target_state))
                    elif AutomatonClass is Mdp:
                        state.transitions[in_sym].append((target_state, count / total))
                    elif AutomatonClass is StochasticMealyMachine:
                        state.transitions[in_sym].append((target_state, out_sym, count / total))

        return AutomatonClass(initial_state, list(state_map.values()))

    def visualize(self, path : Union[str, pathlib.Path], output_behavior : OutputBehavior = "mealy", format : str = "dot", engine ="dot", *,
                  state_label : StateFunction = None, state_color : StateFunction = None,
                  trans_label : TransitionFunction = None, trans_color : TransitionFunction = None,
                  state_props : Dict[str,StateFunction] = None,
                  trans_props : Dict[str,TransitionFunction] = None,
                  node_naming : StateFunction = None):

        # handle default parameters
        if output_behavior not in ["moore", "mealy", None]:
            raise ValueError(f"Invalid OutputBehavior {output_behavior}")
        if state_props is None:
            state_props = dict()
        if trans_props is None:
            trans_props = dict()
        if state_label is None:
            if output_behavior == "moore":
                def state_label(node : Node) : return f'{node.get_prefix_output()} {node.count()}'
            else:
                def state_label(node : Node) : return f'{sum(t.count for _, t in node.transition_iterator())}'
        if trans_label is None and "label" not in trans_props:
            if output_behavior == "moore":
                def trans_label(node : Node, in_sym, out_sym) : return f'{in_sym} [{node.transitions[in_sym][out_sym].count}]'
            else:
                def trans_label(node : Node, in_sym, out_sym) : return f'{in_sym} / {out_sym} [{node.transitions[in_sym][out_sym].count}]'
        if state_color is None:
            def state_color(x) : return "black"
        if trans_color is None:
            def trans_color(x, y, z) : return "black"
        if node_naming is None:
            node_dict = dict()
            def node_naming(node : Node):
                if node not in node_dict:
                    node_dict[node] = f"s{len(node_dict)}"
                return node_dict[node]
        state_props = {"label" : state_label, "color" : state_color , "fontcolor" : state_color, **state_props}
        trans_props = {"label" : trans_label, "color" : trans_color, "fontcolor" : trans_color, **trans_props}

        # create new graph
        graph = pydot.Dot('automaton', graph_type='digraph')

        #graph.add_node(pydot.Node(str(self.prefix), label=state_label(self)))
        nodes = self.get_all_nodes()

        # add nodes
        for node in nodes:
            arg_dict = {key : fun(node) for key, fun in state_props.items()}
            graph.add_node(pydot.Node(node_naming(node), **arg_dict))

        # add transitions
        for node in nodes:
            for in_sym, options in node.transitions.items():
                for out_sym, c in options.items():
                    arg_dict = {key : fun(node, in_sym, out_sym) for key, fun in trans_props.items()}
                    graph.add_edge(pydot.Edge(node_naming(node), node_naming(c.target), **arg_dict))

        # add initial state
        # TODO maybe add option to parameterize this
        graph.add_node(pydot.Node('__start0', shape='none', label=''))
        graph.add_edge(pydot.Edge('__start0', node_naming(self), label=''))

        file_ext = format
        if format == 'dot':
            format = 'raw'
        if format == 'raw':
            file_ext = 'dot'
        graph.write(path=str(path) + "." + file_ext, prog=engine, format=format)

    def add_data(self, data):
        for seq in data:
            self.add_trace(seq)

    def add_trace(self, data : IOTrace):
        curr_node : Node = self
        for in_sym, out_sym in data:
            transitions = curr_node.get_or_create_transitions(in_sym)
            info = transitions.get(out_sym)
            if info is None:
                node = Node((in_sym, out_sym), curr_node)
                transitions[out_sym] = TransitionInfo(node, 1, node, 1)
            else:
                info.count += 1
                info.original_count += 1
                node = info.target
            curr_node = node

    def add_example(self, data : IOExample):
        # TODO add support for example based algorithms
        raise NotImplementedError()

    @staticmethod
    def createPTA(data, output_behavior) -> 'Node':
        if output_behavior == "moore":
            initial_output = data[0][0]
            data = (d[1:] for d in data)
        else:
            initial_output = None
        root_node = Node((None, initial_output), None)
        root_node.add_data(data)
        return root_node

    def is_locally_deterministic(self):
        return all(len(item) == 1 for item in self.transitions.values())

    def is_deterministic(self):
        return all(node.is_locally_deterministic() for node in self.get_all_nodes())

    def deterministic_compatible(self, other : 'Node'):
        common_keys = filter(lambda key: key in self.transitions.keys(), other.transitions.keys())
        return all(list(self.transitions[key].keys()) == list(other.transitions[key].keys()) for key in common_keys)

    def is_moore(self):
        output_dict = dict()
        for node in self.get_all_nodes():
            for (in_sym, out_sym), transition in node.transition_iterator():
                child = transition.target
                if child in output_dict.keys() and output_dict[child] != out_sym:
                    return False
                else:
                    output_dict[child] = out_sym
        return True

    def moore_compatible(self, other : 'Node'):
        return self.get_prefix_output() == other.get_prefix_output()

    def local_log_likelihood_contribution(self):
        llc = 0
        for in_sym, trans in self.transitions.items():
            total_count = 0
            for out_sym, info in trans.items():
                total_count += info.count
                llc += info.count * math.log(info.count)
            if total_count != 0:
              llc -= total_count * math.log(total_count)
        return llc

    def count(self):
        return sum(trans.count for _, trans in self.transition_iterator())

class Partition:
    def __init__(self, a_nodes = None, b_nodes = None):
        self.a_nodes : Set[Node] = a_nodes or set()
        self.b_nodes : Set[Node] = b_nodes or set()

    def __len__(self):
        return len(self.a_nodes) + len(self.b_nodes)

class FoldResult:
    def __init__(self):
        self.partitions : Set[Partition] = set()
        self.counter_examples = []

class StopMode(Enum):
    Stop = 0
    StopExploration = 1
    Continue = 2

def try_fold(a : 'Node', b : 'Node',
             compat : Callable[[Node, Node, Dict[Node, Partition]], Any] = None,
             stop_on_error : Callable[[Any], StopMode] = None
             ) -> FoldResult:
    """
    compute the partitions of two automata that result from grouping two nodes.
    supports custom compatibility criteria for early stopping in case of incompatibility and/or reporting mismatches.
    """
    compat = compat or (lambda a,b,c : True)
    stop_on_error = stop_on_error or (lambda err : StopMode.Stop)
    result = FoldResult()

    partition_map : Dict[Node, Partition] = dict()
    q : deque[Tuple[Node, Node, list]] = deque([(a, b, [])])
    pair_set : Set[Tuple[Node, Node]] = {(a, b)}

    while len(q) != 0:
        a, b, prefix = q.popleft()

        # get partitions
        a_part = partition_map.get(a)
        b_part = partition_map.get(b)

        if a_part is None:
            a_part = Partition({a}, set())
            partition_map[a] = a_part
            result.partitions.add(a_part)
        if b_part is None:
            b_part = Partition(set(), {b})
            partition_map[b] = b_part
            result.partitions.add(b_part)
        if a_part is b_part:
            continue

        # determine compatibility
        compat_result = compat(a, b, partition_map)
        if compat_result is not True:
            if compat_result is not False:
                error = (compat_result, prefix)
            else:
                error = prefix
            result.counter_examples.append(error)
            stop_mode = stop_on_error(compat_result)
            if stop_mode == StopMode.Stop:
                break
            elif stop_mode == StopMode.StopExploration:
                continue
            elif stop_mode == StopMode.Continue:
                pass # Just continue as if nothing happened.

        # merge partitions
        if len(a_part) < len(b_part):
            other_part, part = a_part, b_part
        else:
            part, other_part = a_part, b_part

        part.a_nodes.update(other_part.a_nodes)
        part.b_nodes.update(other_part.b_nodes)

        for node in itertools.chain(other_part.a_nodes, other_part.b_nodes):
            partition_map[node] = part

        result.partitions.remove(other_part)

        # add children to work queue
        for in_sym, a_trans in a.transitions.items():
            b_trans = b.transitions.get(in_sym)
            if b_trans is None:
                continue
            for out_sym, a_next in a_trans.items():
                b_next = b_trans.get(out_sym)
                if b_next is None or (a_next.target, b_next.target) in pair_set:
                    continue
                pair_set.add((a_next.target, b_next.target))
                q.append((a_next.target, b_next.target, prefix + [(in_sym, out_sym)]))

    return result