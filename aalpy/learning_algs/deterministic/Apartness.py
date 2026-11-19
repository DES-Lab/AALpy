# Apartness checks and witness computation between observation-tree nodes (used by L# and related algorithms).
from collections import deque


class Apartness:
    """
    Collection of static helper methods for checking apartness (a distinguishable-by-some-suffix relation) between
    nodes of an observation tree, and between observation-tree nodes and hypothesis states.
    """

    @staticmethod
    def compute_witness(state1, state2, ob_tree) -> list | None:
        """
        Finds a distinguishing sequence between two states if they are apart based on the observation tree.

        :param state1: First observation-tree node (MealyNode or MooreNode).
        :param state2: Second observation-tree node (MealyNode or MooreNode).
        :param ObservationTree ob_tree: Observation tree the states belong to.
        :return list | None: Distinguishing input sequence, or None if the states are not apart.
        """
        if ob_tree.automaton_type == 'mealy':
            state1_destination = Apartness._show_states_are_apart_mealy(
                state1, state2, ob_tree.alphabet)
        else:
            state1_destination = Apartness._show_states_are_apart_moore(
                state1, state2, ob_tree.alphabet)
        if not state1_destination:
            return None
        return ob_tree.get_transfer_sequence(state1, state1_destination)

    @staticmethod
    def states_are_apart(state1, state2, ob_tree) -> bool:
        """
        Checks if two states are apart by checking any output difference in the observation tree.

        :param state1: First observation-tree node (MealyNode or MooreNode).
        :param state2: Second observation-tree node (MealyNode or MooreNode).
        :param ObservationTree ob_tree: Observation tree the states belong to.
        :return bool: True if the states are apart, False otherwise.
        """
        if ob_tree.automaton_type == 'mealy':
            return Apartness._show_states_are_apart_mealy(state1, state2, ob_tree.alphabet) is not None
        else:
            return Apartness._show_states_are_apart_moore(state1, state2, ob_tree.alphabet) is not None

    @staticmethod
    def _show_states_are_apart_mealy(first, second, alphabet: list):
        """
        Identifies if two Mealy observation-tree nodes can be distinguished by any input-output pair in the
        provided alphabet.

        :param first: First observation-tree node (MealyNode).
        :param second: Second observation-tree node (MealyNode).
        :param list alphabet: Input alphabet.
        :return MealyNode | None: The node reached from `first` where a difference was observed, or None.
        """
        pairs = deque([(first, second)])

        while pairs:
            first_node, second_node = pairs.popleft()
            for input_val in alphabet:
                first_output = first_node.get_output(input_val)
                second_output = second_node.get_output(input_val)

                if first_output is not None and second_output is not None:
                    if first_output != second_output:
                        return first_node.get_successor(input_val)

                    pairs.append((first_node.get_successor(
                        input_val), second_node.get_successor(input_val)))

        return None

    @staticmethod
    def _show_states_are_apart_moore(first, second, alphabet: list):
        """
        Identifies if two Moore/DFA observation-tree nodes can be distinguished by any input-output pair in the
        provided alphabet.

        :param first: First observation-tree node (MooreNode).
        :param second: Second observation-tree node (MooreNode).
        :param list alphabet: Input alphabet.
        :return MooreNode | None: The node where a difference in output was observed, or None.
        """
        pairs = deque([(first, second)])

        while pairs:
            first_node, second_node = pairs.popleft()
            if first_node is not None and second_node is not None:
                first_output = first_node.output
                second_output = second_node.output
                if first_output != second_output:
                    return first_node

                for input_val in alphabet:
                    pairs.append((first_node.get_successor(
                        input_val), second_node.get_successor(input_val)))

        return None

    @staticmethod
    def compute_witness_in_tree_and_hypothesis_states(ob_tree, ob_tree_state, hyp_state) -> list | None:
        """
        Determines if the observation tree and the hypothesis are distinguishable based on their state outputs.

        :param ObservationTree ob_tree: Observation tree.
        :param ob_tree_state: Observation-tree node to compare from.
        :param AutomatonState hyp_state: Hypothesis state to compare from.
        :return list | None: Distinguishing input sequence, or None if not distinguishable.
        """
        if ob_tree.automaton_type == 'mealy':
            return Apartness.compute_witness_in_tree_and_hypothesis_states_mealy(ob_tree, ob_tree_state, hyp_state)
        else:
            return Apartness.compute_witness_in_tree_and_hypothesis_states_moore(ob_tree, ob_tree_state, hyp_state)

    @staticmethod
    def compute_witness_in_tree_and_hypothesis_states_mealy(ob_tree, ob_tree_state, hyp_state) -> list | None:
        """
        Determines if the observation tree and the Mealy hypothesis are distinguishable based on their state outputs.

        :param ObservationTree ob_tree: Observation tree.
        :param ob_tree_state: Observation-tree node (MealyNode) to compare from.
        :param MealyState hyp_state: Hypothesis state to compare from.
        :return list | None: Distinguishing input sequence, or None if not distinguishable.
        """
        pairs = deque([(ob_tree_state, hyp_state)])

        while pairs:
            tree_state, hyp_state = pairs.popleft()

            for input_val in ob_tree.alphabet:
                tree_output = tree_state.get_output(input_val)

                if tree_output is not None and input_val in hyp_state.output_fun:
                    hyp_output = hyp_state.output_fun[input_val]
                    if tree_output != hyp_output:
                        tree_dest = tree_state.get_successor(input_val)
                        return ob_tree.get_transfer_sequence(ob_tree_state, tree_dest)

                    pairs.append((tree_state.get_successor(
                        input_val), hyp_state.transitions[input_val]))

        return None

    @staticmethod
    def compute_witness_in_tree_and_hypothesis_states_moore(ob_tree, ob_tree_state, hyp_state) -> list | None:
        """
        Determines if the observation tree and the Moore/DFA hypothesis are distinguishable based on their state
        outputs.

        :param ObservationTree ob_tree: Observation tree.
        :param ob_tree_state: Observation-tree node (MooreNode) to compare from.
        :param AutomatonState hyp_state: Hypothesis state to compare from.
        :return list | None: Distinguishing input sequence, or None if not distinguishable.
        """
        pairs = deque([(ob_tree_state, hyp_state)])

        while pairs:
            tree_state, hyp_state = pairs.popleft()
            if (tree_state is not None) and (hyp_state is not None):
                tree_output = tree_state.output
                if ob_tree.automaton_type == 'dfa':
                    hyp_output = hyp_state.is_accepting
                else:
                    hyp_output = hyp_state.output

                if tree_output != hyp_output:
                    return ob_tree.get_transfer_sequence(ob_tree_state, tree_state)

                for input_val in ob_tree.alphabet:
                    if input_val in hyp_state.transitions:
                        pairs.append((tree_state.get_successor(
                            input_val), hyp_state.transitions[input_val]))

        return None
