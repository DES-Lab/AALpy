# Tree structure used by ONFSM learning algorithms to keep track of all observed input/output traces.
from collections import defaultdict
from typing import Any

from aalpy.automata import Onfsm


class Node:
    """
    Single node of a :class:`TraceTree`, representing one observed input/output pair.
    """

    __slots__ = ['output', 'children', 'parent', 'frequency_counter']

    def __init__(self, output: Any) -> None:
        """
        Creates a trace tree node.

        :param Any output: Output associated with this node.
        """
        self.output = output
        self.children: dict[Any, list['Node']] = defaultdict(list)
        self.parent: Node | None = None

        # frq counter
        self.frequency_counter = 0

    def get_child(self, inp: Any, out: Any) -> 'Node | None':
        """
        Looks up the child reached via the given input/output pair.

        :param Any inp: Input.
        :param Any out: Output.
        :return Node | None: Matching child node, or None if not found.
        """
        return next((child for child in self.children[inp] if child.output == out), None)

    def get_prefix(self) -> tuple:
        """
        Reconstructs the sequence of outputs leading from the root to this node.

        :return tuple: Sequence of outputs on the path from the root to this node.
        """
        prefix = ()
        curr_node = self
        while curr_node.parent is not None:
            prefix = (curr_node.output,) + prefix
            curr_node = curr_node.parent
        return prefix


class TraceTree:
    """
    Tree used for keeping track of seen observations.
    """

    def __init__(self) -> None:
        """
        Creates an empty trace tree containing only the root node.
        """
        self.root_node = Node(None)
        self.curr_node: Node | None = None

    def reset(self) -> None:
        """
        Resets the current node cursor back to the root node.
        """
        self.curr_node = self.root_node

    def add_to_tree(self, inp: Any, out: Any) -> None:
        """
        Adds new element to tree and makes it the current node.

        :param Any inp: Input.
        :param Any out: Output.
        """
        if inp not in self.curr_node.children.keys() or \
                out not in {child.output for child in self.curr_node.children[inp]}:
            node = Node(out)
            self.curr_node.children[inp].append(node)
            node.parent = self.curr_node

        self.curr_node = self.curr_node.get_child(inp, out)
        self.curr_node.frequency_counter += 1

    def add_trace(self, inputs: tuple, outputs: tuple) -> None:
        """
        Adds a whole input/output trace to the tree, starting from the root.

        :param tuple inputs: Sequence of inputs.
        :param tuple outputs: Sequence of outputs.
        """
        self.reset()
        for i, o in zip(inputs, outputs):
            self.add_to_tree(i, o)

    def get_to_node(self, inputs: tuple, outputs: tuple) -> Node | None:
        """
        Follows the path described by inputs and outputs and returns the node which is reached.

        :param tuple inputs: Sequence of inputs.
        :param tuple outputs: Sequence of outputs.
        :return Node | None: Node that is reached when following the given input and output through the tree,
            or None if the path does not exist.
        """
        curr_node = self.root_node
        for i, o in zip(inputs, outputs):
            node = curr_node.get_child(i, o)
            if node is None:
                return None
            curr_node = node

        return curr_node

    def get_all_traces(self, prefix: tuple[tuple, tuple], e: tuple) -> list[tuple]:
        """
        Follows `prefix` (an (inputs, outputs) pair) through the tree, and for the reached node returns all
        traces of outputs corresponding to the input sequence `e`.

        :param tuple[tuple, tuple] prefix: (inputs, outputs) pair identifying the starting node.
        :param tuple e: Sequence of inputs to be traced from the starting node.
        :return list[tuple]: Traces of outputs corresponding to the input sequence given by e.
        """

        if not prefix or not e:
            return []

        curr_node = self.root_node
        for i, o in zip(prefix[0], prefix[1]):
            curr_node = curr_node.get_child(i, o)
            if curr_node is None:
                return []

        queue = [(curr_node, 0)]
        reached_nodes = []
        while queue:
            node, depth = queue.pop(0)
            if depth == len(e):
                reached_nodes.append(node)
            else:
                children_with_same_input = node.children[e[depth]]
                for c in children_with_same_input:
                    queue.append((c, depth + 1))

        cell = [node.get_prefix()[-len(e):] for node in reached_nodes]
        return cell

    def get_table(self, s: list, e: list) -> dict:
        """
        Generates a table from the tree.

        :param list s: Rows from S, S_dot_A, or both which should be presented in the table.
        :param list e: E set (suffixes).
        :return dict: A table in a format that can be used for printing.
        """
        result = {}
        for prefix in s:
            result[prefix] = {}

            for inp in e:
                result[prefix][inp] = self.get_all_traces(prefix, inp)

        return result

    def find_cex_in_cache(self, hypothesis: Onfsm) -> tuple[list, list] | None:
        """
        Searches the cached traces for a counterexample against the given hypothesis, without querying the SUL.

        :param Onfsm hypothesis: Current hypothesis.
        :return tuple[list, list] | None: (inputs, outputs) counterexample, or None if none is found in the cache.
        """

        queue = [(self.root_node, tuple())]
        while queue:
            curr_node, path = queue.pop(0)

            if path:
                hypothesis.reset_to_initial()
                inputs, outputs = [], []
                for i, o in zip(path[0::2], path[1::2]):
                    inputs.append(i)
                    outputs.append(o)
                    out = hypothesis.step_to(i, o)
                    if out is None:
                        return inputs, outputs
            for inp in curr_node.children.keys():
                children = curr_node.children[inp]
                for child in children:
                    # if curr_node.frequency_counter[(inp, child_out)] >= threshold:
                    queue.append((child, path + (inp, child.output)))

        return None

    def get_s_e_sampling_frequency(self, prefix: tuple[tuple, tuple], suffix: tuple) -> int:
        """
        Counts how many times the path described by `prefix` followed by `suffix` has been observed.

        :param tuple[tuple, tuple] prefix: (inputs, outputs) pair identifying the starting node.
        :param tuple suffix: Sequence of inputs to be traced from the starting node.
        :return int: Number of times the given path has been sampled.
        """
        sampling_frequency = 0
        curr_node = self.root_node
        for i, o in zip(prefix[0], prefix[1]):
            curr_node = curr_node.get_child(i, o)
            if curr_node is None:
                return 0

        queue = [(curr_node, 0)]
        while queue:
            node, depth = queue.pop(0)
            children_with_same_input = node.children[suffix[depth]]
            if depth == len(suffix) - 1:
                for c in children_with_same_input:
                    sampling_frequency += c.frequency_counter
            else:
                for c in children_with_same_input:
                    queue.append((c, depth + 1))

        return sampling_frequency

    def get_sampling_distributions(self, prefix: tuple[tuple, tuple], input_from_alphabet: Any) -> dict:
        """
        Computes the empirical output probability distribution observed after `prefix` on a given input.

        :param tuple[tuple, tuple] prefix: (inputs, outputs) pair identifying the starting node.
        :param Any input_from_alphabet: Single input from the alphabet.
        :return dict: Map from observed output to its empirical probability.
        """
        sampling_distribution = {}
        curr_node = self.root_node
        for i, o in zip(prefix[0], prefix[1]):
            curr_node = curr_node.get_child(i, o)

        children = curr_node.children[input_from_alphabet]
        sampling_sum = sum(c.frequency_counter for c in children)
        for c in children:
            sampling_distribution[c.output] = c.frequency_counter / sampling_sum

        return sampling_distribution
