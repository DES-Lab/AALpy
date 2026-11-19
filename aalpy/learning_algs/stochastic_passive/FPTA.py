# Frequency Prefix Tree Acceptor (FPTA) data structure used by Alergia/IOAlergia as the initial hypothesis.
from functools import total_ordering
from typing import Any


@total_ordering
class AlergiaPtaNode:
    """
    Single node of the frequency prefix tree acceptor (FPTA). Keeps both the current (mutable) and original
    (immutable) children/input frequencies, the latter being used for statistical compatibility checks.
    """

    __slots__ = ['prefix', 'output', 'input_frequency', 'children', 'original_input_frequency',
                 'original_children', 'state_id', 'children_prob']

    def __init__(self, output: Any, prefix: tuple) -> None:
        """
        Creates a new FPTA node.

        :param Any output: Output associated with this node.
        :param tuple prefix: Sequence of inputs (and/or outputs, depending on automaton type) that lead to this node
            from the root.
        """
        self.prefix = prefix
        self.output = output
        # mutable values
        self.input_frequency = dict()
        self.children = dict()
        # immutable values used for statistical computability check
        self.original_input_frequency = dict()
        self.original_children = dict()
        # # for visualization
        self.state_id = None
        self.children_prob = None

    def successors(self) -> list['AlergiaPtaNode']:
        """
        Returns the (mutable) children of this node.

        :return list[AlergiaPtaNode]: List of successor nodes.
        """
        return list(self.children.values())

    def get_inputs(self) -> set:
        """
        Returns the set of inputs observed in the mutable input frequency map.

        :return set: Set of inputs.
        """
        return {i for i, _ in self.input_frequency.keys()}

    def get_input_frequency(self, target_input: Any) -> int:
        """
        Computes the total frequency of a given input, summed over all outputs.

        :param Any target_input: Input whose frequency should be computed.
        :return int: Total observed frequency of the input.
        """
        return sum(freq for (i, _), freq in self.input_frequency.items() if i == target_input)

    def get_output_frequencies(self, target_input: Any) -> dict:
        """
        Returns the frequency of each output observed for a given input.

        :param Any target_input: Input for which output frequencies should be computed.
        :return dict: Mapping of output to observed frequency.
        """
        return {o: freq for (i, o), freq in self.input_frequency.items() if i == target_input}

    def get_immutable_inputs(self) -> set:
        """
        Returns the set of inputs observed in the original (immutable) children map.

        :return set: Set of inputs.
        """
        return {i for i, _ in self.original_children.keys()}

    def get_immutable_input_frequency(self, target_input: Any) -> int:
        """
        Computes the total original frequency of a given input, summed over all outputs.

        :param Any target_input: Input whose frequency should be computed.
        :return int: Total original frequency of the input.
        """
        return sum(freq for (i, _), freq in self.original_input_frequency.items() if i == target_input)

    def get_original_output_frequencies(self, target_input: Any) -> dict:
        """
        Returns the original frequency of each output observed for a given input.

        :param Any target_input: Input for which output frequencies should be computed.
        :return dict: Mapping of output to original observed frequency.
        """
        return {o: freq for (i, o), freq in self.original_input_frequency.items() if i == target_input}

    def __lt__(self, other: 'AlergiaPtaNode') -> bool:
        return (len(self.prefix), self.prefix) < (len(other.prefix), other.prefix)

    def __le__(self, other: 'AlergiaPtaNode') -> bool:
        return self < other or self == other

    def __eq__(self, other: 'AlergiaPtaNode') -> bool:
        return self.prefix == other.prefix


def create_fpta(data: list, automaton_type: str) -> AlergiaPtaNode:
    """
    Builds the frequency prefix tree acceptor (FPTA) from a data set of observed traces.

    :param list data: Data set of traces, format depends on automaton_type (see run_Alergia for details).
    :param str automaton_type: Either 'mc', 'mdp', or 'smm'.
    :return AlergiaPtaNode: Root node of the constructed FPTA.
    """
    # in case of SMM, there is no initial input
    seq_iter_index = 0 if automaton_type == 'smm' else 1

    initial_output = None if automaton_type == 'smm' else data[0][0]

    root_node = AlergiaPtaNode(initial_output, ())

    for seq in data:
        if automaton_type != 'smm' and seq[0] != root_node.output:
            print('All sequences passed to Alergia should have the same initial output!')
            assert False

        curr_node = root_node

        for el in seq[seq_iter_index:]:
            if el not in curr_node.children:
                out = None
                if automaton_type == 'mc':
                    out = el
                elif automaton_type == 'mdp':
                    out = el[1]

                reached_node = AlergiaPtaNode(out, curr_node.prefix + (el,))
                curr_node.children[el] = reached_node
                curr_node.original_children[el] = reached_node

                curr_node.input_frequency[el] = 0
                curr_node.original_input_frequency[el] = 0

            curr_node.input_frequency[el] += 1
            curr_node.original_input_frequency[el] += 1

            curr_node = curr_node.children[el]

    return root_node
