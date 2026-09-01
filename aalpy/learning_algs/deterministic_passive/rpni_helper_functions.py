# Helper data structures and functions shared by the RPNI-family passive learning algorithms.
import pickle
from functools import total_ordering
from typing import Any


@total_ordering
class RpniNode:
    """
    Single node of the prefix tree acceptor (PTA) used during RPNI-style state merging.
    """

    __slots__ = ['output', 'children', 'prefix', "type"]

    def __init__(self, output: Any = None, children: dict | None = None, automaton_type: str = 'moore') -> None:
        """
        Creates a PTA node.

        :param Any output: Output of the node. For 'mealy' automata this is a dict mapping input to output.
        :param dict | None children: Map from input symbol to child RpniNode.
        :param str automaton_type: Either 'dfa', 'moore', or 'mealy'.
        """
        if output is None and automaton_type == 'mealy':
            output = dict()
        if children is None:
            children = dict()
        self.output = output
        self.children = children
        self.prefix = ()
        self.type = automaton_type

    def shallow_copy(self) -> 'RpniNode':
        """
        Creates a shallow copy of this node, copying the children map (and output dict for 'mealy').

        :return RpniNode: The shallow copy.
        """
        output = self.output if self.type != 'mealy' else dict(self.output)
        return RpniNode(output, dict(self.children), self.type)

    def copy(self) -> 'RpniNode':
        """
        Creates a deep copy of this node and its whole subtree.

        :return RpniNode: The deep copy.
        """
        return pickle.loads(pickle.dumps(self, -1))

    def __lt__(self, other: 'RpniNode') -> bool:
        """
        Compares nodes by prefix length, used to keep the blue/red state lists sorted.

        :param RpniNode other: Node to compare against.
        :return bool: True if this node's prefix is shorter than other's.
        """
        return len(self.prefix) < len(other.prefix)
        # return (len(self.prefix), self.prefix) < (len(other.prefix), other.prefix)

    def __eq__(self, other: 'RpniNode') -> bool:
        """
        Compares nodes by prefix.

        :param RpniNode other: Node to compare against.
        :return bool: True if both nodes have the same prefix.
        """
        return self.prefix == other.prefix

    def __hash__(self) -> int:
        """
        :return int: Identity-based hash of the node.
        """
        return id(self)  # TODO This is a hack

    def compatible_outputs(self, other: 'RpniNode') -> bool:
        """
        Only allow merging of states that have same output(s).

        :param RpniNode other: Node to check compatibility against.
        :return bool: True if the outputs of both nodes are compatible.
        """
        # None is compatible with everything
        if self.type != 'mealy':
            return self.output == other.output or self.output is None or other.output is None
        else:
            red_io = {i: o for i, o in self.output.items()}
            blue_io = {i: o for i, o in other.output.items()}
            for common_i in set(red_io.keys()).intersection(blue_io.keys()):
                if red_io[common_i] != blue_io[common_i]:
                    return False
        return True

    def get_child_by_prefix(self, prefix: tuple) -> 'RpniNode':
        """
        Follows a sequence of input symbols from this node and returns the reached node.

        :param tuple prefix: Sequence of input symbols to follow.
        :return RpniNode: The node reached after following the prefix.
        """
        node = self
        for symbol in prefix:
            node = node.children[symbol]
        return node


def check_sequence(root_node: RpniNode, seq: list, automaton_type: str) -> bool:
    """
    Checks whether each sequence in the dataset is valid in the current automaton.

    :param RpniNode root_node: Root node of the (partial) automaton represented as a PTA.
    :param list seq: Sequence of (input, output) pairs (optionally preceded by an initial output for non-mealy
        automata) to validate.
    :param str automaton_type: Either 'dfa', 'moore', or 'mealy'.
    :return bool: True if the sequence is consistent with the automaton, False otherwise.
    """
    curr_node = root_node
    # Check initial output for Moore machines and the like
    if automaton_type != 'mealy' and len(seq) != 0:
        if seq[0] is not None and seq[0] != root_node.output:
            return False
        seq = seq[1:]
    for i, o in seq:
        if automaton_type == 'mealy':
            if i not in curr_node.output or o is not None and curr_node.output[i] != o:
                return False
            curr_node = curr_node.children[i]
        else:
            # For dfa and moore, check if outputs are the same, iff output in test data is concrete (not None)
            curr_node = curr_node.children[i]
            if o is not None and curr_node.output != o:
                return False
    return True


def createPTA(data: list, automaton_type: str) -> RpniNode | None:
    """
    Constructs a prefix tree acceptor (PTA) from the provided data.

    :param list data: Sequence of (input sequence, label) pairs.
    :param str automaton_type: Either 'dfa', 'moore', or 'mealy'.
    :return RpniNode | None: The root node of the constructed PTA, or None if the data is non-deterministic.
    """
    data.sort(key=lambda x: len(x[0]))

    root_node = RpniNode(automaton_type=automaton_type)
    for seq, label in data:
        curr_node = root_node
        for idx, symbol in enumerate(seq):
            if symbol not in curr_node.children.keys():
                node = RpniNode(automaton_type=automaton_type)
                node.prefix = curr_node.prefix + (symbol,)
                curr_node.children[symbol] = node

            if automaton_type == 'mealy' and idx == len(seq) - 1:
                if symbol not in curr_node.output:
                    curr_node.output[symbol] = label
                if curr_node.output[symbol] != label:
                    return None
            curr_node = curr_node.children[symbol]
        if automaton_type == 'moore' or automaton_type == 'dfa':
            if curr_node.output is None:
                curr_node.output = label
            if curr_node.output != label:
                return None

    return root_node


def extract_unique_sequences(root_node: RpniNode, automaton_type: str) -> list[list]:
    """
    Extracts, for every leaf of the PTA, the unique sequence of (input, output) pairs leading to it.

    :param RpniNode root_node: Root node of the PTA.
    :param str automaton_type: Either 'dfa', 'moore', or 'mealy'.
    :return list[list]: List of sequences, one per leaf node of the PTA.
    """
    def get_leaf_nodes(root: RpniNode) -> list[RpniNode]:
        leaves = []

        def _get_leaf_nodes(node: RpniNode) -> None:
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
        seq = [] if automaton_type == 'mealy' else [root_node.output]
        curr_node = root_node
        for i in node.prefix:
            if automaton_type == 'mealy':
                seq.append((i, curr_node.output.get(i)))
                curr_node = curr_node.children[i]
            else:
                curr_node = curr_node.children[i]
                seq.append((i, curr_node.output))
        paths.append(seq)

    return paths


def to_automaton(red: list[RpniNode], automaton_type: str) -> Any:
    """
    Converts a list of merged (red) PTA nodes into the corresponding automaton.

    :param list[RpniNode] red: List of red (final) states of the PTA, in prefix-length order (the first entry is
        the initial state).
    :param str automaton_type: Either 'dfa', 'moore', or 'mealy'.
    :return Any: The constructed Dfa, MooreMachine, or MealyMachine.
    """
    from aalpy.automata import DfaState, Dfa, MooreMachine, MooreState, MealyMachine, MealyState

    if automaton_type == 'dfa':
        state, automaton = DfaState, Dfa
    elif automaton_type == 'moore':
        state, automaton = MooreState, MooreMachine
    else:
        state, automaton = MealyState, MealyMachine

    initial_state = None
    prefix_state_map = {}
    for i, r in enumerate(red):
        if automaton_type == 'moore' or automaton_type == 'dfa':
            # make sure all None states (incomplete information to deduce output) are set to False for DFAs
            if automaton_type == 'dfa' and r.output is None:
                r.output = False
            prefix_state_map[r.prefix] = state(f's{i}', r.output)
        else:
            prefix_state_map[r.prefix] = state(f's{i}')
        if i == 0:
            initial_state = prefix_state_map[r.prefix]

    for r in red:
        for i, c in r.children.items():
            if automaton_type == 'moore' or automaton_type == 'dfa':
                prefix_state_map[r.prefix].transitions[i] = prefix_state_map[c.prefix]
            else:
                prefix_state_map[r.prefix].transitions[i] = prefix_state_map[c.prefix]
                prefix_state_map[r.prefix].output_fun[i] = r.output[i] if i in r.output else None

    return automaton(initial_state, list(prefix_state_map.values()))


def visualize_pta(root_node: RpniNode, path: str = 'pta.pdf') -> None:
    """
    Visualizes the PTA and writes the resulting graph to a PDF file.

    :param RpniNode root_node: Root node of the PTA to visualize.
    :param str path: Output file path for the generated PDF.
    """
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
                graph.add_node(Node(str(c.prefix), label=f'{c.output}'))
            graph.add_edge(Edge(str(curr.prefix), str(c.prefix), label=f'{i}'))
            if c.prefix not in visited:
                queue.append(c)
            visited.add(c.prefix)

    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', str(root_node.prefix), label=''))

    graph.write(path=path, format='pdf')
