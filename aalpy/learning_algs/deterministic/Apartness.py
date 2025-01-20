from collections import deque

class Apartness:
    @staticmethod
    def compute_witness(state1, state2, ob_tree):
        """
        Finds a distinguishing sequence between two states if they are apart based on the observation tree
        """
        state1_destination = Apartness._show_states_are_apart(state1, state2, ob_tree.alphabet)
        if not state1_destination:
            return
        return ob_tree.get_transfer_sequence(state1, state1_destination)

    @staticmethod
    def states_are_apart(state1, state2, ob_tree):
        """
        Checks if two states are apart by checking any output difference in the observation tree
        """
        return Apartness._show_states_are_apart(state1, state2, ob_tree.alphabet) is not None

    @staticmethod
    def _show_states_are_apart(first, second, alphabet):
        """
        Identifies if two states can be distinguished by any input-output pair in the provided alphabet
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
                    
                    pairs.append((first_node.get_successor(input_val), second_node.get_successor(input_val)))
        
        return None
    
    @staticmethod
    def compute_witness_in_tree_and_hypothesis(ob_tree, hypothesis):
        """
        Finds a distinguishing sequence between the observation tree and the hypothesis if they differ
        """
        tree_destination = Apartness._show_states_are_apart_in_tree_and_hypothesis(hypothesis, ob_tree)
        if not tree_destination:
            return
        return ob_tree.get_transfer_sequence(ob_tree.root, tree_destination)

    @staticmethod
    def _show_states_are_apart_in_tree_and_hypothesis(hypothesis, ob_tree):
        """
        Determines if the observation tree and the hypothesis are distinguishable based on their state outputs
        """
        pairs = deque([(ob_tree.root, hypothesis.initial_state)])

        while pairs:
            tree_state, hyp_state = pairs.popleft()

            for input_val in ob_tree.alphabet:
                tree_output = tree_state.get_output(input_val)
                hyp_output = hyp_state.output_fun[input_val]

                if tree_output is not None and hyp_output is not None:
                    if tree_output != hyp_output: 
                        return tree_state.get_successor(input_val)
                    
                    pairs.append((tree_state.get_successor(input_val), hyp_state.transitions[input_val]))

        return None