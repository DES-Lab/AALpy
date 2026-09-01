# Classic (non-generalized) RPNI passive learning algorithm implementation.
import time
from bisect import insort
from typing import Any

from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import to_automaton, createPTA, \
    check_sequence, extract_unique_sequences, RpniNode


class ClassicRPNI:
    """
    Classic RPNI implementation that performs state merging directly on a copy of the PTA for every merge attempt.
    """

    def __init__(self, data: list, automaton_type: str, print_info: bool = True) -> None:
        """
        Creates a ClassicRPNI instance and constructs the prefix tree acceptor (PTA) from the data.

        :param list data: Sequence of (input sequence, label) pairs.
        :param str automaton_type: Either 'dfa', 'mealy', or 'moore'.
        :param bool print_info: Whether to print learning progress and runtime information.
        """
        self.data = data
        self.automaton_type = automaton_type
        self.print_info = print_info

        pta_construction_start = time.time()
        self.root_node = createPTA(data, automaton_type)
        self.test_data = extract_unique_sequences(self.root_node, automaton_type)

        if self.print_info:
            print(f'PTA Construction Time: {round(time.time() - pta_construction_start, 2)}')

    def run_rpni(self) -> Any:
        """
        Runs the classic RPNI state-merging procedure and constructs the resulting automaton.

        :return Any: The learned Dfa, MooreMachine, or MealyMachine.
        """
        start_time = time.time()

        red = [self.root_node]
        blue = list(red[0].children.values())
        while blue:
            lex_min_blue = min(list(blue))
            merged = False

            for red_state in red:
                if not red_state.compatible_outputs(lex_min_blue):
                    continue
                merge_candidate = self._merge(red_state, lex_min_blue, copy_nodes=True)
                if self._compatible(merge_candidate):
                    self._merge(red_state, lex_min_blue)
                    merged = True
                    break

            if not merged:
                insort(red, lex_min_blue)
                if self.print_info:
                    print(f'\rCurrent automaton size: {len(red)}', end="")

            blue.clear()
            for r in red:
                for c in r.children.values():
                    if c not in red:
                        blue.append(c)

        if self.print_info:
            print(f'\nRPNI Learning Time: {round(time.time() - start_time, 2)}')
            print(f'RPNI Learned {len(red)} state automaton.')

        assert sorted(red, key=lambda x: len(x.prefix)) == red
        return to_automaton(red, self.automaton_type)

    def _compatible(self, root_node: RpniNode) -> bool:
        """
        Check if current model is compatible with the data.

        :param RpniNode root_node: Root node of the model to check.
        :return bool: True if the model is compatible with all test data, False otherwise.
        """
        for sequence in self.test_data:
            if not check_sequence(root_node, sequence, automaton_type=self.automaton_type):
                return False
        return True

    def _merge(self, red_node: RpniNode, lex_min_blue: RpniNode, copy_nodes: bool = False) -> RpniNode:
        """
        Merge two states and return the root node of resulting model.

        :param RpniNode red_node: Red state to merge into.
        :param RpniNode lex_min_blue: Blue state to merge.
        :param bool copy_nodes: Whether to perform the merge on a copy of the PTA (used for compatibility checks)
            rather than in place.
        :return RpniNode: Root node of the (possibly copied) model after merging.
        """
        root_node = self.root_node.copy() if copy_nodes else self.root_node
        lex_min_blue = lex_min_blue.copy() if copy_nodes else lex_min_blue

        red_node_in_tree = root_node
        for p in red_node.prefix:
            red_node_in_tree = red_node_in_tree.children[p]

        to_update = root_node
        for p in lex_min_blue.prefix[:-1]:
            to_update = to_update.children[p]

        to_update.children[lex_min_blue.prefix[-1]] = red_node_in_tree

        if self.automaton_type != 'mealy':
            self._fold(red_node_in_tree, lex_min_blue)
        else:
            self._fold_mealy(red_node_in_tree, lex_min_blue)

        return root_node

    def _fold(self, red_node: RpniNode, blue_node: RpniNode) -> None:
        """
        Recursively folds a blue node's subtree into a red node's subtree for non-mealy automata.

        :param RpniNode red_node: Red node to fold into.
        :param RpniNode blue_node: Blue node to fold.
        """
        # Change the output of red only to concrete output, ignore None
        red_node.output = blue_node.output if blue_node.output is not None else red_node.output

        for i in blue_node.children.keys():
            if i in red_node.children.keys():
                self._fold(red_node.children[i], blue_node.children[i])
            else:
                red_node.children[i] = blue_node.children[i]

    def _fold_mealy(self, red_node: RpniNode, blue_node: RpniNode) -> None:
        """
        Recursively folds a blue node's subtree into a red node's subtree for mealy automata.

        :param RpniNode red_node: Red node to fold into.
        :param RpniNode blue_node: Blue node to fold.
        """
        for i, o in blue_node.output.items():
            red_node.output[i] = o

        for i in blue_node.children.keys():
            if i in red_node.children.keys():
                self._fold_mealy(red_node.children[i], blue_node.children[i])
            else:
                red_node.children[i] = blue_node.children[i]
