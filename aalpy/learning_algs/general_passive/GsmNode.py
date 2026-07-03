import functools
import pathlib
from collections import defaultdict
from functools import total_ordering
from typing import Dict, Any, List, Tuple, Iterable, Callable, Union, TypeVar, Iterator, Optional, Sequence, Generic
import pydot

from aalpy.automata import StochasticMealyMachine, StochasticMealyState, MooreState, MooreMachine, NDMooreState, \
    NDMooreMachine, Mdp, MdpState, MealyMachine, MealyState, Onfsm, OnfsmState
from aalpy.base import Automaton
from aalpy.learning_algs.general_passive.IOHandler import (
    IOHandler,
    NoIOHandler,
    StochasticData,
    CountData,
)

Key = TypeVar("Key")
Val = TypeVar("Val")
T = TypeVar("T")

OutputBehavior = str
OutputBehaviorRange = ["moore", "mealy"]

TransitionBehavior = str
TransitionBehaviorRange = ["deterministic", "nondeterministic", "stochastic"]

DataFormat = str
DataFormatRange = ["io_traces", "labeled_sequences", "traces", "tree"]

IOPair = Tuple[Any, Any]
IOTrace = Sequence[IOPair]
IOExample = Tuple[Sequence[Any], Any]

StateFunction = Callable[['GsmNode'], str]
TransitionFunction = Callable[['GsmNode', Any, Any], str]

unknown_output = None  # can be set to a special value if required


def intersection_iterator(a: Dict[Key, Val], b: Dict[Key, Val], sort_by_length=False) -> Iterator[Tuple[Key, Val, Val]]:
    missing = object()
    if sort_by_length and len(b) < len(a):
        for key, b_val in b.items():
            a_val = a.get(key, missing)
            if a_val is missing:
                continue
            yield key, a_val, b_val
    else:
        for key, a_val in a.items():
            b_val = b.get(key, missing)
            if b_val is missing:
                continue
            yield key, a_val, b_val


def union_iterator(a: Dict[Key, Val], b: Dict[Key, Val], default: Val = None) -> Iterator[Tuple[Key, Val, Val]]:
    for key, a_val in a.items():
        b_val = b.get(key, default)
        yield key, a_val, b_val
    for key, b_val in b.items():
        if key in a:
            continue
        a_val = a.get(key, default)
        yield key, a_val, b_val


# TODO reuse in RPNI
def detect_data_format(data, check_consistency=False, guess=False):
    # The different data formats are
    # - "tree": a tree-shaped automaton provided as a GsmNode
    # - "io_traces": either
    #   - Moore traces [[o, (i,o), (i,o), ...], ...]
    #   - Mealy traces [[(i,o), (i,o), ...], ...]
    # - "labeled_sequences": [([i, i, ...], o), ...]
    # - "traces": [[o, o, ...], ...]

    if isinstance(data, GsmNode):
        return "tree"

    accepted_types = (Tuple, List)

    # mapping data formats to compatibility criteria
    check_dict = dict(
        io_traces=lambda obj: len(obj) <= 1 or all(isinstance(o, accepted_types) and len(o) == 2 for o in obj[1:]),
        labeled_sequences=lambda obj: len(obj) == 2 and isinstance(obj[0], accepted_types),
    )
    accept_dict = {k: True for k in check_dict}

    if not isinstance(data, accepted_types):
        raise ValueError("wrong input format. expected tuple or list.")
    if len(data) == 0:
        return "io_traces"

    accepted_formats = list(accept_dict.keys())
    for data_point in data:
        if not isinstance(data_point, accepted_types):
            raise ValueError("wrong input format. expected tuple or list.")
        for k, check in check_dict.items():
            accept_dict[k] &= check(data_point)
        accepted_formats = [k for k, v in accept_dict.items() if v]
        if len(accepted_formats) == 1 and not check_consistency:
            return accepted_formats[0]
        if len(accepted_formats) == 0:
            return "traces" # default to traces
            #raise ValueError("invalid or inconsistent data. no options left")
    if len(accepted_formats) != 1 and not guess:
        raise ValueError("ambiguous data format. data format needs to be specified explicitly.")
    return accepted_formats[0]

# TODO add custom pickling code that flattens the Node structure in order to circumvent running into recursion issues for large models
@total_ordering
class GsmNode(Generic[T]):
    """
    Generic class for observably deterministic automata.

    The prefix is given as (minimal) list of IO pairs leading to that state.
    We assume an initial transition to the initial state, which has to be reflected in the prefix.
    This way, the output of the initial state for Moore machines can be encoded in its prefix.

    Transition count is preferred over state count as it allows to easily count transitions for non-tree-shaped automata
    """
    __slots__ = ['transitions', 'predecessor', 'prefix_access_pair', 'data']

    def __init__(self, prefix_access_pair, predecessor: 'GsmNode[T]' = None, data: T = None): # TODO (data-ext) check all invocations
        # TODO try single dict
        self.transitions: defaultdict[Any, Dict[Any, GsmNode[T]]] = defaultdict(dict)
        self.predecessor: GsmNode = predecessor
        self.prefix_access_pair = prefix_access_pair
        self.data = data

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

    def get_prefix_input(self):
        return self.prefix_access_pair[0]

    def resolve_unknown_prefix_output(self, value):
        p_in, p_out = self.prefix_access_pair
        if p_out is unknown_output:
            self.prefix_access_pair = (p_in, value)

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

    def get_root(self):
        current = self
        while current.predecessor:
            current = current.predecessor
        return current

    def get_or_create_transitions(self, in_sym) -> Dict[Any, 'GsmNode[T]']:
        t = self.transitions.get(in_sym)
        if t is None:
            t = dict()
            self.transitions[in_sym] = t
        return t

    def transition_iterator(self) -> Iterable[Tuple[Any, Any, 'GsmNode[T]']]:
        for in_sym, transitions in self.transitions.items():
            for out_sym, node in transitions.items():
                yield in_sym, out_sym, node

    def shallow_copy(self, data_handler: IOHandler) -> 'GsmNode[T]':
        node = GsmNode(self.prefix_access_pair, self.predecessor, data_handler.copy(self.data))
        for in_sym, t in self.transitions.items():
            node.transitions[in_sym] = t.copy()
        return node

    def get_by_prefix(self, seq: IOTrace) -> Optional['GsmNode[T]']:
        node: GsmNode = self
        for in_sym, out_sym in seq:
            if in_sym is None:  # ignore initial transition of Node.get_prefix()
                continue
            trans = node.transitions.get(in_sym)
            if trans is None:
                return None
            node = trans.get(out_sym)
            if node is None:
                return None
        return node

    def get_all_nodes(self) -> List['GsmNode[T]']:
        result = [self]
        backing_set = {self}
        for state in result:
            for _, _, child in state.transition_iterator():
                if child not in backing_set:
                    backing_set.add(child)
                    result.append(child)
        return result

    def is_tree(self):
        q: List['GsmNode'] = [self]
        backing_set = {self}
        while len(q) != 0:
            current = q.pop(0)
            for _, _, child in current.transition_iterator():
                if child in backing_set:
                    return False
                q.append(child)
                backing_set.add(child)
        return True

    def to_automaton(self, output_behavior: OutputBehavior, transition_behavior: TransitionBehavior,
                     check_behavior=True, set_prefix=False) -> Automaton:
        nodes = self.get_all_nodes()

        if check_behavior:
            if output_behavior == "moore" and not self.is_moore():
                raise ValueError("Tried to obtain Moore machine from non-Moore structure")
            if transition_behavior == "deterministic" and not self.is_deterministic():
                raise ValueError("Tried to obtain deterministic automaton from non-deterministic structure")

        type_dict = {
            ("moore", "deterministic"): (MooreMachine, MooreState),
            ("moore", "nondeterministic"): (NDMooreMachine, NDMooreState),
            ("moore", "stochastic"): (Mdp, MdpState),
            ("mealy", "deterministic"): (MealyMachine, MealyState),
            ("mealy", "nondeterministic"): (Onfsm, OnfsmState),
            ("mealy", "stochastic"): (StochasticMealyMachine, StochasticMealyState),
        }

        automaton_class, state_class = type_dict[(output_behavior, transition_behavior)]

        # create states
        state_map = dict()
        for i, node in enumerate(nodes):
            state_id = f's{i}'
            if output_behavior == "mealy":
                state = state_class(state_id)
            elif output_behavior == "moore":
                state = state_class(state_id, node.get_prefix_output())
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
            if automaton_class in [Mdp, StochasticMealyMachine]:
                if not isinstance(node.data, StochasticData):
                    raise TypeError(f"No probability in information available for {automaton_class.__name__}")
                prob_info = node.data.get_probabilities()
            for in_sym, transitions in node.transitions.items():
                for out_sym, target_node in transitions.items():
                    target_state = state_map[target_node]
                    if automaton_class is MooreMachine:
                        state.transitions[in_sym] = target_state
                    elif automaton_class is MealyMachine:
                        state.transitions[in_sym] = target_state
                        state.output_fun[in_sym] = out_sym
                    elif automaton_class is NDMooreMachine:
                        state.transitions[in_sym].append(target_state)
                    elif automaton_class is Onfsm:
                        state.transitions[in_sym].append((out_sym, target_state))
                    elif automaton_class is Mdp:
                        state.transitions[in_sym].append((target_state, prob_info[in_sym][out_sym]))
                    elif automaton_class is StochasticMealyMachine:
                        state.transitions[in_sym].append((target_state, out_sym, prob_info[in_sym][out_sym]))

        return automaton_class(initial_state, list(state_map.values()))

    def visualize(self, path: Union[str, pathlib.Path], output_behavior: OutputBehavior = "mealy", format: str = "dot",
                  engine="dot", *,
                  state_label: StateFunction = None, state_color: StateFunction = None,
                  trans_label: TransitionFunction = None, trans_color: TransitionFunction = None,
                  state_props: Dict[str, StateFunction] = None,
                  trans_props: Dict[str, TransitionFunction] = None,
                  node_naming: StateFunction = None):

        # handle default parameters
        if output_behavior not in ["moore", "mealy", None]:
            raise ValueError(f"Invalid OutputBehavior {output_behavior}")
        if state_props is None:
            state_props = dict()
        if trans_props is None:
            trans_props = dict()
        if state_label is None:
            def state_label(node: GsmNode):
                label_parts = []
                if output_behavior == "moore":
                    label_parts.append(str(node.get_prefix_output()))
                if isinstance(node.data, CountData):
                    label_parts.append(str(node.data.count()))
                return " ".join(label_parts)
        if trans_label is None and "label" not in trans_props:
            def trans_label(node: GsmNode, in_sym, out_sym):
                label_parts = []
                if output_behavior == "moore":
                    label_parts.append(str(in_sym))
                else:
                    label_parts.append(f'{in_sym} / {out_sym}')
                if isinstance(node.data, CountData):
                    label_parts.append(f'[{node.data.transition_count[in_sym][out_sym]}]')
                return " ".join(label_parts)
        if state_color is None:
            def state_color(x): return "black"
        if trans_color is None:
            def trans_color(x, y, z): return "black"
        if node_naming is None:
            node_dict = dict()

            def node_naming(node: GsmNode):
                if node not in node_dict:
                    node_dict[node] = f"s{len(node_dict)}"
                return node_dict[node]
        state_props = {"label": state_label, "color": state_color, "fontcolor": state_color, **state_props}
        trans_props = {"label": trans_label, "color": trans_color, "fontcolor": trans_color, **trans_props}

        # create new graph
        graph = pydot.Dot('automaton', graph_type='digraph')

        # graph.add_node(pydot.Node(str(self.prefix), label=state_label(self)))
        nodes = self.get_all_nodes()

        # add nodes
        for node in nodes:
            arg_dict = {key: fun(node) for key, fun in state_props.items()}
            graph.add_node(pydot.Node(node_naming(node), **arg_dict))

        # add transitions
        for node in nodes:
            for in_sym, options in node.transitions.items():
                for out_sym, c in options.items():
                    arg_dict = {key: fun(node, in_sym, out_sym) for key, fun in trans_props.items()}
                    graph.add_edge(pydot.Edge(node_naming(node), node_naming(c), **arg_dict))

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

    def make_input_complete(self, ic_mode=None) -> List[Tuple['GsmNode', Any, Any]]:
        ic_modes = ["self-loop", "sink-state", "root"]
        if ic_mode is None:
            ic_mode = ic_modes[0]
        if ic_mode not in ic_modes:
            raise ValueError(f"Invalid ic_mode {ic_mode}. Should be one of {ic_modes}")

        all_nodes = self.get_all_nodes()
        inputs = {in_sym for node in all_nodes for in_sym in node.transitions}
        missing_trans = []
        for node in all_nodes:
            for in_sym in inputs:
                transitions = node.transitions[in_sym]
                if len(transitions) == 0:
                    out_sym = node.prefix_access_pair[1]
                    missing_trans.append((node, in_sym, out_sym))
                    if ic_mode == "self-loop":
                        successor = node
                    elif ic_mode == "sink-state":
                        raise NotImplementedError()
                    elif ic_mode == "root":
                        successor = self
                    transitions[out_sym] = successor
        return missing_trans

    def add_trace(self, trace: IOTrace, data_handler: IOHandler[T]):
        curr_node: GsmNode = self
        for in_value, out_value in trace:
            in_sym, out_sym = data_handler.abstract(in_value, out_value)
            transitions = curr_node.transitions[in_sym]
            node = transitions.get(out_sym)
            if node is None:
                node = GsmNode((in_sym, out_sym), curr_node, data_handler.init_data())
                transitions[out_sym] = node
            data_handler.aggregate_data(curr_node, in_value, out_value, node)
            curr_node = node

    def add_labeled_sequence(self, example: IOExample, data_handler: IOHandler[T] = None):
        inputs, output = example
        curr_node: GsmNode = self
        in_sym = None

        if not isinstance(data_handler, NoIOHandler):
            raise NotImplementedError("Data handling is not supported for learning from labeled sequences")

        # step through inputs and add transitions
        for in_value in inputs:
            in_sym, out_sym = data_handler.abstract(in_value, None)
            transitions = curr_node.transitions[in_sym]
            successors = list(transitions.values())
            if len(successors) == 0:
                node = GsmNode((in_sym, unknown_output), curr_node)
                transitions[unknown_output] = node
            elif len(successors) == 1:
                node = successors[0]
            else:
                # This should never happen
                raise ValueError("Nondeterminism encountered for GSM with labeled_sequences. not supported")
            data_handler.aggregate_data(curr_node, in_value, None, node)
            curr_node = node

        # set last output
        curr_node.resolve_unknown_prefix_output(output)
        pred = curr_node.predecessor
        if pred:
            transitions = pred.transitions[in_sym]
            if unknown_output in transitions:
                transitions[output] = transitions.pop(unknown_output)
            if output not in transitions:
                raise ValueError("nondeterminism encountered for GSM with labeled_sequences. not supported")

    @staticmethod
    def createPTA(data, output_behavior, data_format=None, data_handler: IOHandler[T] = None) -> 'GsmNode':
        if data_format is None:
            data_format = detect_data_format(data)
        if data_format not in DataFormatRange:
            raise ValueError(f"invalid data format {data_format}. should be in {DataFormatRange}")

        data_handler.init(data, output_behavior, data_format)

        if data_format == "tree":
            if not data.is_tree():
                raise ValueError("provided automaton is not a tree")
            return data
        # TODO extract method for replaying data on dot model
        root_node = GsmNode((None, unknown_output), None, data_handler.init_data())
        if data_format == "labeled_sequences":
            for example in data:
                root_node.add_labeled_sequence(example, data_handler)
        if data_format == "io_traces" or data_format == "traces":
            if output_behavior == "moore":
                root_node.prefix_access_pair = data_handler.abstract(None, data[0][0])
                initial_output_symbol = root_node.prefix_access_pair[1]

                for trace in data:
                    initial_output = trace[0]
                    _, ios = data_handler.abstract(None, initial_output)
                    if ios != initial_output_symbol:
                        raise ValueError("expect unique initial output symbol for Moore behavior")
                    data_handler.aggregate_data(None, None, initial_output, root_node)

                data = (d[1:] for d in data)
            for trace in data:
                if data_format == "traces":
                    trace = (("step", t) for t in trace)
                root_node.add_trace(trace, data_handler)
        return root_node

    def is_locally_deterministic(self):
        return all(len(item) == 1 for item in self.transitions.values())

    def is_deterministic(self):
        return all(node.is_locally_deterministic() for node in self.get_all_nodes())

    def deterministic_compatible(self, other: 'GsmNode'):
        for _, trans_self, trans_other in intersection_iterator(self.transitions, other.transitions):
            if unknown_output in trans_self or unknown_output in trans_other:
                continue
            if trans_self.keys() != trans_other.keys():
                return False
        return True

    def is_moore(self):
        for node in self.get_all_nodes():
            for in_sym, out_sym, next_node in node.transition_iterator():
                child_output = next_node.get_prefix_output()
                if out_sym is not unknown_output and child_output != out_sym:
                    return False
        return True

    def moore_compatible(self, other: 'GsmNode'):
        so = self.get_prefix_output()
        oo = other.get_prefix_output()
        return so == oo or so is unknown_output or oo is unknown_output

    default_order = functools.cmp_to_key(lambda a, b: -1 if a < b else 1)
