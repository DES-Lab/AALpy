from abc import ABC, abstractmethod
from collections import defaultdict


class AutomatonState(ABC):

    def __init__(self, state_id):
        """
        Single state of an automaton. Each state consists of a state id, a dictionary of transitions, where the keys are
        inputs and the values are the corresponding target states, and a prefix that leads to the state from the initial
        state.

        Args:

            state_id(Any): used for graphical representation of the state. A good practice is to keep it unique.

        """
        self.state_id = state_id
        self.transitions = dict()
        self.prefix = None

    def get_diff_state_transitions(self) -> list:
        """
        Returns a list of transitions that lead to new states, not same-state transitions.
        """
        transitions = []
        for trans, state in self.transitions.items():
            if state != self:
                transitions.append(trans)
        return transitions

    def get_same_state_transitions(self) -> list:
        """
        Get all transitions that lead to the same state (self loops).
        """
        dst = self.get_diff_state_transitions()
        all_trans = set(self.transitions.keys())
        return [t for t in all_trans if t not in dst]


class Automaton(ABC):
    """
    Abstract class representing an automaton.
    """

    def __init__(self, initial_state, states: list):
        """
        Args:

            initial_state (AutomatonState): initial state of the automaton
            states (list) : list containing all states of the automaton

        """
        self.initial_state = initial_state
        self.states = states
        self.characterization_set: list = []
        self.current_state = initial_state
        self.size = len(self.states)

    def reset_to_initial(self):
        """
        Resets the current state of the automaton to the initial state
        """
        self.current_state = self.initial_state

    @abstractmethod
    def step(self, letter):
        """
        Performs a single step on the automaton changing its current state.

        Args:

            letter: element of the input alphabet to be executed on the system under learning

        Returns:

            Output produced when executing the input letter from the current state

        """
        pass

    def is_input_complete(self) -> bool:
        """
        Check whether all states have defined transition for all inputs
        :return: true if automaton is input complete

        Returns:

            True if input complete, False otherwise

        """
        alphabet = set(self.get_input_alphabet())
        for state in self.states:
            if set(state.transitions.keys()) != alphabet:
                return False
        return True

    def get_input_alphabet(self) -> list:
        """
        Returns the input alphabet
        """
        alphabet = list()
        for s in self.states:
            for i in s.transitions.keys():
                if i not in alphabet:
                    alphabet.append(i)
        return list(alphabet)

    def get_state_by_id(self, state_id) -> AutomatonState:
        for state in self.states:
            if state.state_id == state_id:
                return state

        return None

    def __str__(self):
        """
        :return: A string representation of the automaton
        """
        from aalpy.utils import save_automaton_to_file
        return save_automaton_to_file(self, path='learnedModel', file_type='string', round_floats=2)

    def execute_sequence(self, origin_state, seq):
        self.current_state = origin_state
        return [self.step(s) for s in seq]


class DeterministicAutomaton(Automaton):

    @abstractmethod
    def step(self, letter):
        pass

    def get_shortest_path(self, origin_state: AutomatonState, target_state: AutomatonState) -> tuple:
        """
        Breath First Search over the automaton

        Args:

            origin_state (AutomatonState): state from which the BFS will start
            target_state (AutomatonState): state that will be reached with the return value

        Returns:

            sequence of inputs that lead from origin_state to target state

        """
        if origin_state not in self.states or target_state not in self.states:
            raise SystemExit("State not in the automaton.")

        explored = []
        queue = [[origin_state]]

        if origin_state == target_state:
            return ()

        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in explored:
                neighbours = node.transitions.values()
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    # return path if neighbour is goal
                    if neighbour == target_state:
                        acc_seq = new_path[:-1]
                        inputs = []
                        for ind, state in enumerate(acc_seq):
                            inputs.append(next(key for key, value in state.transitions.items()
                                               if value == new_path[ind + 1]))
                        return tuple(inputs)

                # mark node as explored
                explored.append(node)
        return ()

    def is_strongly_connected(self) -> bool:
        """
        Check whether the automaton is strongly connected,
        meaning that every state can be reached from every other state.

        Returns:

            True if strongly connected, False otherwise

        """
        import itertools

        state_comb_list = itertools.permutations(self.states, 2)
        for state_comb in state_comb_list:
            if not self.get_shortest_path(state_comb[0], state_comb[1]):
                return False
        return True

    def output_step(self, state, letter):
        """
            Given an input letter, compute the output response from a given state.
            Args:
                state: state from which the output response shall be computed
                letter: an input letter from the alphabet

            Returns: the single-step output response

        """
        state_save = self.current_state
        self.current_state = state
        output = self.step(letter)
        self.current_state = state_save
        return output

    def find_distinguishing_seq(self, state1, state2):
        """
        A BFS to determine an input sequence that distinguishes two states in the automaton, i.e., a sequence such that
        the output response from the given states is different. In a minimal automaton, this function always returns a
        sequence different from None
        Args:
            state1: first state
            state2: second state to distinguish

        Returns: an input sequence distinguishing two states, or None if the states are equivalent

        """
        visited = set()
        to_explore = [(state1, state2, [])]
        alphabet = self.get_input_alphabet()
        while to_explore:
            (curr_s1, curr_s2, prefix) = to_explore.pop(0)
            visited.add((curr_s1, curr_s2))
            for i in alphabet:
                o1 = self.output_step(curr_s1, i)
                o2 = self.output_step(curr_s2, i)
                new_prefix = prefix + [i]
                if o1 != o2:
                    return new_prefix
                else:
                    next_s1 = curr_s1.transitions[i]
                    next_s2 = curr_s2.transitions[i]
                    if (next_s1, next_s2) not in visited:
                        to_explore.append((next_s1, next_s2, new_prefix))

        raise SystemExit('Distinguishing sequence could not be computed (Non-canonical automaton).')

    def compute_output_seq(self, state, sequence):
        """
        Given an input sequence, compute the output response from a given state.
        Args:
            state: state from which the output response shall be computed
            sequence: an input sequence over the alphabet

        Returns: the output response

        """
        state_save = self.current_state
        output = self.execute_sequence(state, sequence)
        self.current_state = state_save
        return output

    def compute_characterization_set(self, char_set_init=None, online_suffix_closure=True, split_all_blocks=True):
        """
        Computation of a characterization set, that is, a set of sequences that can distinguish all states in the
        automation. The implementation follows the approach for finding multiple preset diagnosing experiments described
        by Arthur Gill in "Introduction to the Theory of Finite State Machines".
        Some optional parameterized adaptations, e.g., for computing suffix-closed sets target the application in
        L*-based learning and conformance testing.
        The function only works for minimal automata.
        Args:
            char_set_init: a list of sequence that will be included in the characterization set, e.g., the input
                        alphabet. A empty sequance is added to this list when using automata with state labels
                        (DFA and Moore)
            online_suffix_closure: if true, ensures suffix closedness of the characterization set at every computation
                                step
            split_all_blocks: if false, the computation follows the original tree-based strategy, where newly computed
                        sequences are only checked on a subset of the states to be distinguished
                        if true, sequences are used to distinguish all states, yielding a potentially smaller set, which
                        is useful for conformance testing and learning

        Returns: a characterization set

        """
        from copy import copy

        blocks = list()
        blocks.append(copy(self.states))
        char_set = [] if not char_set_init else char_set_init
        if char_set_init:
            for seq in char_set_init:
                blocks = self._split_blocks(blocks, seq)

        while True:
            # Given a partition (of states), this function returns a block with at least two elements.
            try:
                block_to_split = next(filter(lambda b: len(b) > 1, blocks))
            except StopIteration:
                block_to_split = None

            if not block_to_split:
                break
            split_state1 = block_to_split[0]
            split_state2 = block_to_split[1]
            dist_seq = self.find_distinguishing_seq(split_state1, split_state2)
            assert ((not split_all_blocks) or (dist_seq not in char_set))

            # in L*-based learning, we use suffix-closed column labels, so it makes sense to use a suffix-closed
            # char set in this context
            if online_suffix_closure:
                dist_seq_closure = [tuple(dist_seq[len(dist_seq) - i - 1:]) for i in range(len(dist_seq))]
            else:
                dist_seq_closure = [tuple(dist_seq)]

            # the standard approach described by Gill, computes a sequence that splits one block and really only splits
            # one block, that is, it is only applied to the states in said block
            # in L*-based learning we combine every prefix with every, therefore it makes sense to apply the sequence
            # on all blocks and split all
            if split_all_blocks:
                for seq in dist_seq_closure:
                    # seq may be in char_set if we do the closure on the fly
                    if seq in char_set:
                        continue
                    char_set.append(seq)
                    blocks = self._split_blocks(blocks, seq)
            else:
                blocks.remove(block_to_split)
                new_blocks = [block_to_split]
                for seq in dist_seq_closure:
                    char_set.append(seq)
                    new_blocks = self._split_blocks(new_blocks, seq)
                for new_block in new_blocks:
                    blocks.append(new_block)

        char_set = list(set(char_set))
        return char_set

    def _split_blocks(self, blocks, seq):
        """
        Refines a partition of states (blocks) using the output response to a given input sequence seq.
        Args:
            blocks: a partition of states
            seq: an input sequence

        Returns: a refined partition of states

        """
        new_blocks = []
        for block in blocks:
            block_after_split = defaultdict(list)
            for state in block:
                output_seq = tuple(self.compute_output_seq(state, seq))
                block_after_split[output_seq].append(state)
            for new_block in block_after_split.values():
                new_blocks.append(new_block)
        return new_blocks
