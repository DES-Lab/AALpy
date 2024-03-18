import copy
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, TypeVar, Generic, List


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
        self.transitions = None
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


AutomatonStateType = TypeVar("AutomatonStateType", bound=AutomatonState)

OutputType = TypeVar("OutputType")
InputType = TypeVar("InputType")


class Automaton(ABC, Generic[AutomatonStateType]):
    """
    Abstract class representing an automaton.
    """

    def __init__(self, initial_state: AutomatonStateType, states: List[AutomatonStateType]):
        """
        Args:

            initial_state (AutomatonState): initial state of the automaton
            states (list) : list containing all states of the automaton

        """
        self.initial_state: AutomatonStateType = initial_state
        self.states: List[AutomatonStateType] = states
        self.characterization_set: list = []
        self.current_state: AutomatonStateType = initial_state

    @property
    def size(self):
        return len(self.states)

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

    # returns a list which is input alphabet, or a sevpa alphabet in case of VPAs
    def get_input_alphabet(self):
        """
        Returns the input alphabet
        """
        alphabet = list()
        for s in self.states:
            for i in s.transitions.keys():
                if i not in alphabet:
                    alphabet.append(i)
        return list(alphabet)

    def get_state_by_id(self, state_id) -> Union[AutomatonStateType, None]:
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

    def make_input_complete(self, missing_transition_go_to='self_loop'):
        """
        For more details check the implementation of this method in utils.HelperFunctions

        missing_transition_go_to: either 'self_loop' or 'sink_state'.
        """
        from aalpy.utils.HelperFunctions import make_input_complete
        make_input_complete(self, missing_transition_go_to)

    def execute_sequence(self, origin_state, seq):
        self.current_state = origin_state
        return [self.step(s) for s in seq]

    def save(self, file_path='LearnedModel', file_type='dot'):
        from aalpy.utils import save_automaton_to_file
        save_automaton_to_file(self, path=file_path, file_type=file_type)

    def visualize(self, path='LearnedModel', file_type='pdf', display_same_state_transitions=True):
        from aalpy.utils import visualize_automaton
        visualize_automaton(self, path, file_type, display_same_state_transitions)

    @staticmethod
    @abstractmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'Automaton':
        pass

    @abstractmethod
    def to_state_setup(self):
        pass

    def copy(self) -> 'Automaton':
        return self.from_state_setup(self.to_state_setup())

    def __reduce__(self):
        return self.from_state_setup, (self.to_state_setup(),)


class DeterministicAutomaton(Automaton[AutomatonStateType]):

    @abstractmethod
    def step(self, letter):
        pass

    def get_shortest_path(self, origin_state: AutomatonStateType, target_state: AutomatonStateType) -> Union[
        tuple, None]:
        """
        Breath First Search over the automaton to find the shortest path

        Args:

            origin_state (AutomatonState): state from which the BFS will start
            target_state (AutomatonState): state that will be reached with the return value

        Returns:

            sequence of inputs that lead from origin_state to target state, or None if target state is not reachable
            from origin state

        """
        if origin_state not in self.states or target_state not in self.states:
            warnings.warn('Origin or target state not in automaton. Returning None.')
            return None

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

        return None

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
            if self.get_shortest_path(state_comb[0], state_comb[1]) is None:
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

    def find_distinguishing_seq(self, state1, state2, alphabet):
        """
        A BFS to determine an input sequence that distinguishes two states in the automaton, i.e., a sequence such that
        the output response from the given states is different. In a minimal automaton, this function always returns a
        sequence different from None
        Args:
            state1: first state
            state2: second state to distinguish
            alphabet: input alphabet of the automaton

        Returns: an input sequence distinguishing two states, or None if the states are equivalent

        """
        visited = set()
        to_explore = [(state1, state2, [])]
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

        return None

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

    def is_minimal(self):
        if not self.is_input_complete():
            warnings.warn('Minimization of non input complete automata is not yet supported. Returning False.')
            return False
        return self.compute_characterization_set(raise_warning=False) is not None

    def compute_characterization_set(self, char_set_init=None,
                                     online_suffix_closure=True,
                                     split_all_blocks=True,
                                     return_same_states=False,
                                     raise_warning=True):
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
            return_same_states: if True, a single distinguishable pair of states will be returned, or None None if there
                        are no non-distinguishable states
            raise_warning: prints warning message if characterization set cannot be computed

        Returns: a characterization set or None if a non-minimal automaton is passed to the function

        """
        blocks = list()
        blocks.append(copy.copy(self.states))
        char_set = [] if not char_set_init else char_set_init
        if char_set_init:
            for seq in char_set_init:
                blocks = self._split_blocks(blocks, seq)

        alphabet = self.get_input_alphabet()
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
            dist_seq = self.find_distinguishing_seq(split_state1, split_state2, alphabet)
            if dist_seq is None:
                if return_same_states:
                    return split_state1, split_state2

                if raise_warning:
                    warnings.warn("Automaton is non-canonical: could not compute characterization set."
                                  "Returning None.")
                return None

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
        if return_same_states:
            return None, None
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

    def compute_prefixes(self):
        for s in self.states:
            if not s.prefix:
                s.prefix = self.get_shortest_path(self.initial_state, s)

    def minimize(self):
        if not self.is_input_complete():
            warnings.warn('Minimization of non input complete automata is not yet supported.\n Model not minimized.')
            return

        s1, s2 = self.compute_characterization_set(return_same_states=True)
        while s1 and s2:
            for s in self.states:
                for i, new_state in s.transitions.items():
                    if new_state == s2:
                        s.transitions[i] = s1
            self.states.remove(s2)
            s1, s2 = self.compute_characterization_set(return_same_states=True)

        self.compute_prefixes()

    def __eq__(self, other):
        from aalpy.utils import bisimilar
        return bisimilar(self, other)