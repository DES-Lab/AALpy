# Adaptive distinguishing sequence (ADS) construction used to identify frontier states in L#.
from collections import defaultdict
from typing import Any


class AdsNode:
    """
    Single node of an ADS tree, holding the input to send next, a mapping from observed output to the child node,
    and a score describing how well this (sub)tree splits its associated block of states.
    """

    __slots__ = ['input', 'children', 'score']

    def __init__(self, input_val: Any = None, children: dict | None = None, score: float = 0) -> None:
        """
        Creates an ADS node.

        :param Any input_val: Input to send at this node, or None for a leaf.
        :param dict | None children: Map from observed output to child AdsNode.
        :param float score: Score of this (sub)tree.
        """
        self.input = input_val
        self.children = children if children else {}
        self.score = score

    @staticmethod
    def create_leaf() -> 'AdsNode':
        """
        Creates a leaf node (no input, no children).

        :return AdsNode: The created leaf node.
        """
        return AdsNode()

    def get_input(self) -> Any:
        """
        :return Any: Input to be sent at this node.
        """
        return self.input

    def get_child_node(self, output: Any) -> 'AdsNode | None':
        """
        Looks up the child node reached after observing a given output.

        :param Any output: Observed output.
        :return AdsNode | None: Child node for this output, or None if it does not exist.
        """
        if output in self.children:
            return self.children[output]
        return None

    def get_score(self) -> float:
        """
        :return float: Score of this (sub)tree.
        """
        return self.score


class Ads:
    """
    Adaptive distinguishing sequence for a block of observation-tree states. Builds a decision tree of inputs that
    incrementally splits the block based on observed outputs, and can be walked input-by-input using next_input.
    """

    def __init__(self, ob_tree, current_block: list) -> None:
        """
        Constructs the ADS tree for a block of observation-tree nodes.

        :param ObservationTree ob_tree: Observation tree the block belongs to.
        :param list current_block: List of observation-tree nodes to be distinguished.
        """
        self.initial_node = self.construct_ads(ob_tree, current_block)
        self.current_node = self.initial_node

    def get_score(self) -> float:
        """
        :return float: Score of the whole ADS tree.
        """
        return self.initial_node.get_score()

    def construct_ads(self, ob_tree, current_block: list) -> AdsNode:
        """
        Builds the ADS tree recursively by selecting optimal inputs for splitting states.
        For DFA/Moore we have to consider the output for the empty word.

        :param ObservationTree ob_tree: Observation tree the block belongs to.
        :param list current_block: List of observation-tree nodes to be distinguished.
        :return AdsNode: Root node of the constructed ADS tree.
        """
        if ob_tree.automaton_type == 'mealy':
            return self.construct_ads_rec(ob_tree, current_block)
        else:
            if len(current_block) == 1:
                return AdsNode.create_leaf()

            # if none of the nodes in the current block have a successor, we cannot decide a next input
            if not any([True for node in current_block if node.successors is not None]):
                raise RuntimeError("No input available during ADS computation")

            input = tuple()
            empty_part = self.partition_on_output_empty(current_block, ob_tree.automaton_type)
            u_i = sum(len(part) for part in empty_part.values())
            score = 0
            children = {}

            for output, partition in empty_part.items():
                output_score, subtree = self.compute_output_subtree(ob_tree, partition, u_i)
                score += output_score
                children[output] = subtree

            return AdsNode(input, children, score)

    def construct_ads_rec(self, ob_tree, current_block: list) -> AdsNode:
        """
        Builds the ADS tree recursively by selecting optimal inputs for splitting states.

        :param ObservationTree ob_tree: Observation tree the block belongs to.
        :param list current_block: List of observation-tree nodes to be distinguished.
        :return AdsNode: Root node of the constructed (sub)tree.
        """
        if len(current_block) == 1:
            return AdsNode.create_leaf()

        # If none of the nodes in the current block have a successor, we cannot decide a next input
        if not any([True for node in current_block if node.successors is not None]):
            raise RuntimeError("No input available during ADS computation")

        best_input, best_score = self.maximal_base_input(ob_tree.alphabet, current_block, ob_tree.automaton_type)
        best_children = None

        # The maximal apartness pairs is len(current block) - 1, for any current block, immediately return
        if best_score == len(current_block) - 1:
            return AdsNode(best_input, best_children, best_score)

        for input_val in ob_tree.alphabet:
            input_partitions = self.partition_on_output(current_block, input_val, ob_tree.automaton_type)
            u_i = sum(len(part) for part in input_partitions.values())
            input_score = 0
            children = {}

            for output, partition in input_partitions.items():
                output_score, subtree = self.compute_output_subtree(ob_tree, partition, u_i)
                input_score += output_score
                children[output] = subtree

            if input_score > best_score:
                best_score = input_score
                best_input = input_val
                best_children = children
            if best_score == len(current_block) - 1:
                return AdsNode(best_input, best_children, best_score)

        if best_input is None:
            raise RuntimeError("Error during ADS construction")

        return AdsNode(best_input, best_children, best_score)

    # def make_subtree(self, ob_tree, sub_trees, partition):
    #     # Constructs a subtree for a partition and calculates its score
    #     partition_size = len(partition)
    #     child_score = self.construct_ads_rec(ob_tree, partition).get_score()
    #     return self.compute_reg_score(partition_size, sub_trees, child_score)

    def compute_output_subtree(self, ob_tree, partition: list, u_i: int) -> tuple[float, AdsNode]:
        """
        Computes and scores a subtree for a specific output partition.

        :param ObservationTree ob_tree: Observation tree the partition belongs to.
        :param list partition: Nodes belonging to this output partition.
        :param int u_i: Total number of nodes in the parent partition.
        :return tuple[float, AdsNode]: The computed score and the constructed subtree.
        """
        output_subtree = self.construct_ads_rec(ob_tree, partition)
        output_score = self.compute_score(len(partition), u_i, output_subtree.get_score())
        return output_score, output_subtree

    def compute_score(self, u_io: int, u_i: int, child_score: float) -> float:
        """
        Calculates a score based on partition size and subtree characteristics.

        :param int u_io: Number of nodes in the output partition.
        :param int u_i: Total number of nodes in the parent partition.
        :param float child_score: Score of the subtree built from the output partition.
        :return float: Computed score.
        """
        return (u_io * (u_i - u_io + child_score)) / u_i

    def partition_on_output_empty(self, block: list, automaton_type: str) -> defaultdict:
        """
        Partitions states in the block based on their output for the empty word. Only used during the initial call.

        :param list block: Observation-tree nodes to partition.
        :param str automaton_type: Automaton type, one of ['dfa', 'mealy', 'moore'].
        :return defaultdict: Map from output to list of nodes with that output.
        """
        partition = defaultdict(list)
        for node in block:
            output = node.output
            partition[output].append(node)
        return partition

    def partition_on_output(self, block: list, input_val: Any, automaton_type: str) -> defaultdict:
        """
        Partitions states in the block based on their output for a given input.

        :param list block: Observation-tree nodes to partition.
        :param Any input_val: Input on which to partition.
        :param str automaton_type: Automaton type, one of ['dfa', 'mealy', 'moore'].
        :return defaultdict: Map from output to list of successor nodes with that output.
        """
        partition = defaultdict(list)
        for node in block:
            if automaton_type == 'mealy':
                output = node.get_output(input_val)
                if output is not None:
                    successor = node.get_successor(input_val)
                    if successor is not None:
                        partition[output].append(successor)
            else:
                successor = node.get_successor(input_val)
                if successor is not None:
                    output = successor.output
                    if output is not None:
                        partition[output].append(successor)
        return partition

    def next_input(self, prev_output: Any) -> Any:
        """
        Returns the next input based on the previous output and updates the current node.

        :param Any prev_output: Output observed for the previous input, or None on the first call.
        :return Any: Next input to send, or None if the ADS has no further input for this output.
        """
        if prev_output is not None:
            child = self.current_node.get_child_node(prev_output)
            if child is None:
                return None
            self.current_node = child
        return self.current_node.get_input()

    def maximal_base_input(self, alphabet: list, block: list, automaton_type: str) -> tuple[Any, float]:
        """
        Identifies the input with the highest ability to split the state block based on apartness.
        Does not use the recursive part of the formula.

        :param list alphabet: Input alphabet.
        :param list block: Observation-tree nodes to split.
        :param str automaton_type: Automaton type, one of ['dfa', 'mealy', 'moore'].
        :return tuple[Any, float]: The best input found and its score.
        """
        best_input = alphabet[0]
        best_score = 0

        for input_val in alphabet:
            partition = self.partition_on_output(block, input_val, automaton_type)
            u_i = sum(len(part) for part in partition.values())
            score = 0

            for part in partition.values():
                u_io = len(part)
                output_score = (u_io * (u_i - u_io)) / u_i
                score += output_score

            if score > best_score:
                best_score = score
                best_input = input_val

        return best_input, best_score

    def reset_to_root(self) -> None:
        """
        Resets the current ADS node to the initial root node.
        """
        self.current_node = self.initial_node
