# Abstract base classes shared by all automata types: states, deterministic/non-deterministic automata.
import copy
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable
from typing import TypeVar, Generic


class AutomatonState(ABC):
    """
    Abstract single state of an automaton.
    """

    def __init__(self, state_id: Hashable) -> None:
        """
        Single state of an automaton. Each state consists of a state id, a dictionary of transitions, where the keys are
        inputs and the values are the corresponding target states, and a prefix that leads to the state from the initial
        state.

        :param Hashable state_id: Used for graphical representation of the state. A good practice is to keep it
            unique.
        """
        self.state_id = state_id
        self.transitions = None
        self.prefix = None

    def get_diff_state_transitions(self) -> list:
        """
        Returns a list of transitions that lead to new states, not same-state transitions.

        :return list: Transitions that lead to a different state.
        """
        transitions = []
        for trans, state in self.transitions.items():
            if state != self:
                transitions.append(trans)
        return transitions

    def get_same_state_transitions(self) -> list:
        """
        Get all transitions that lead to the same state (self loops).

        :return list: Transitions that lead back to this state.
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

    def __init__(self, initial_state: AutomatonStateType, states: list[AutomatonStateType]) -> None:
        """
        Creates an automaton.

        :param AutomatonState initial_state: Initial state of the automaton.
        :param list[AutomatonStateType] states: List containing all states of the automaton.
        """
        self.initial_state: AutomatonStateType = initial_state
        self.states: list[AutomatonStateType] = states
        self.characterization_set: list = []
        self.current_state: AutomatonStateType = initial_state

    @property
    def size(self) -> int:
        """
        :return int: Number of states in the automaton.
        """
        return len(self.states)

    def reset_to_initial(self) -> None:
        """
        Resets the current state of the automaton to the initial state.
        """
        self.current_state = self.initial_state

    @abstractmethod
    def step(self, letter):
        """
        Performs a single step on the automaton changing its current state.

        :param letter: Element of the input alphabet to be executed on the system under learning.
        :return: Output produced when executing the input letter from the current state.
        """
        pass

    def is_input_complete(self) -> bool:
        """
        Check whether all states have defined transition for all inputs.

        :return bool: True if input complete, False otherwise.
        """
        alphabet = set(self.get_input_alphabet())
        for state in self.states:
            if set(state.transitions.keys()) != alphabet:
                return False
        return True

    def get_input_alphabet(self) -> list:
        """
        Returns the input alphabet.

        :return list: The input alphabet.
        """
        alphabet = list()
        for s in self.states:
            for i in s.transitions.keys():
                if i not in alphabet:
                    alphabet.append(i)
        return list(alphabet)

    def get_state_by_id(self, state_id: Hashable) -> AutomatonStateType | None:
        """
        Looks up a state by its state_id.

        :param Hashable state_id: Identifier of the state to look up.
        :return AutomatonStateType | None: The state with the given id, or None if not found.
        """
        for state in self.states:
            if state.state_id == state_id:
                return state

        return None

    def __str__(self) -> str:
        """
        :return str: A string representation of the automaton.
        """
        from aalpy.utils import save_automaton_to_file
        return save_automaton_to_file(self, path='learnedModel', file_type='string', round_floats=2)

    def make_input_complete(self, missing_transition_go_to: str = 'self_loop') -> None:
        """
        For more details check the implementation of this method in utils.HelperFunctions.

        :param str missing_transition_go_to: Either 'self_loop' or 'sink_state'.
        """
        from aalpy.utils.HelperFunctions import make_input_complete
        make_input_complete(self, missing_transition_go_to)

    def execute_sequence(self, origin_state: AutomatonStateType, seq: list) -> list:
        """
        Executes an input sequence on the automaton starting from a given state.
        Note that execute sequence CHANGES the state!

        :param AutomatonStateType origin_state: State from which the sequence execution starts.
        :param list seq: Input sequence to execute.
        :return list: The output response for the executed sequence.
        """
        self.current_state = origin_state
        return [self.step(s) for s in seq]

    def save(self, file_path: str = 'LearnedModel', file_type: str = 'dot') -> None:
        """
        Saves the automaton to a file.

        :param str file_path: Path (without extension) where the automaton shall be saved.
        :param str file_type: Format to save the automaton in, e.g. 'dot'.
        """
        from aalpy.utils import save_automaton_to_file
        save_automaton_to_file(self, path=file_path, file_type=file_type)

    def visualize(self, path: str = 'LearnedModel', file_type: str = 'pdf',
                  display_same_state_transitions: bool = True) -> None:
        """
        Visualizes the automaton.

        :param str path: Path (without extension) where the visualization shall be saved.
        :param str file_type: Format to render the visualization in, e.g. 'pdf'.
        :param bool display_same_state_transitions: Whether self-loop transitions should be displayed.
        """
        from aalpy.utils import visualize_automaton
        visualize_automaton(self, path, file_type, display_same_state_transitions)

    @staticmethod
    @abstractmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'Automaton':
        """
        Creates an automaton from a state setup dictionary.

        :param dict state_setup: Map from state_id to state configuration.
        :return Automaton: The constructed automaton.
        """
        pass

    @abstractmethod
    def to_state_setup(self):
        """
        Converts the automaton to a state setup dictionary.

        :return dict: Map from state_id to state configuration.
        """
        pass

    def copy(self) -> 'Automaton':
        """
        :return Automaton: A deep copy of the automaton, built via its state setup.
        """
        return self.from_state_setup(self.to_state_setup())

    def __reduce__(self) -> tuple:
        """
        :return tuple: Callable and arguments used to reconstruct the automaton, for pickling.
        """
        return self.from_state_setup, (self.to_state_setup(),)


class DeterministicAutomaton(Automaton[AutomatonStateType]):
    """
    Abstract class representing a deterministic automaton.
    """

    @abstractmethod
    def step(self, letter):
        """
        Performs a single step on the automaton changing its current state.

        :param letter: Element of the input alphabet to be executed on the system under learning.
        :return: Output produced when executing the input letter from the current state.
        """
        pass

    def get_shortest_path(self, origin_state: AutomatonStateType, target_state: AutomatonStateType) -> tuple | None:
        """
        Breath First Search over the automaton to find the shortest path.

        :param AutomatonStateType origin_state: State from which the BFS will start.
        :param AutomatonStateType target_state: State that will be reached with the return value.
        :return tuple | None: Sequence of inputs that lead from origin_state to target state, or None if target
            state is not reachable from origin state.
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

        :return bool: True if strongly connected, False otherwise.
        """
        if not self.states:
            return True

        # strongly connected iff some state reaches all states and is reached by
        # all states; two linear traversals instead of one search per state pair
        all_states = set(self.states)
        root = self.states[0]

        visited = {root}
        stack = [root]
        while stack:
            for successor in stack.pop().transitions.values():
                if successor not in visited and successor in all_states:
                    visited.add(successor)
                    stack.append(successor)
        if visited != all_states:
            return False

        predecessors = {state: [] for state in self.states}
        for state in self.states:
            for successor in state.transitions.values():
                if successor in all_states:
                    predecessors[successor].append(state)

        visited = {root}
        stack = [root]
        while stack:
            for predecessor in predecessors[stack.pop()]:
                if predecessor not in visited:
                    visited.add(predecessor)
                    stack.append(predecessor)
                    
        return visited == all_states

    def output_step(self, state: AutomatonStateType, letter):
        """
        Given an input letter, compute the output response from a given state.

        :param AutomatonStateType state: State from which the output response shall be computed.
        :param letter: An input letter from the alphabet.
        :return: The single-step output response.
        """
        state_save = self.current_state
        self.current_state = state
        output = self.step(letter)
        self.current_state = state_save
        return output

    def find_distinguishing_seq(self, state1: AutomatonStateType, state2: AutomatonStateType, alphabet: list) -> list | None:
        """
        A BFS to determine an input sequence that distinguishes two states in the automaton, i.e., a sequence such that
        the output response from the given states is different. In a minimal automaton, this function always returns a
        sequence different from None.

        :param AutomatonStateType state1: First state.
        :param AutomatonStateType state2: Second state to distinguish.
        :param list alphabet: Input alphabet of the automaton.
        :return list | None: An input sequence distinguishing two states, or None if the states are equivalent.
        """
        if state1 is state2:
            return None
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

    def compute_output_seq(self, state: AutomatonStateType, sequence: list) -> list:
        """
        Given an input sequence, compute the output response from a given state.

        :param AutomatonStateType state: State from which the output response shall be computed.
        :param list sequence: An input sequence over the alphabet.
        :return list: The output response.
        """
        state_save = self.current_state
        output = self.execute_sequence(state, sequence)
        self.current_state = state_save
        return output

    def is_minimal(self) -> bool:
        """
        Checks whether the automaton is minimal, i.e., whether a characterization set can be computed for it.

        :return bool: True if the automaton is minimal, False otherwise.
        """
        if not self.is_input_complete():
            warnings.warn('Minimization of non input complete automata is not yet supported. Returning False.')
            return False
        return self.compute_characterization_set(raise_warning=False) is not None

    def compute_characterization_set(self, char_set_init: list | None = None,
                                     online_suffix_closure: bool = True,
                                     split_all_blocks: bool = True,
                                     return_same_states: bool = False,
                                     raise_warning: bool = True) -> list | tuple | None:
        """
        Computation of a characterization set, that is, a set of sequences that can distinguish all states in the
        automation. The implementation follows the approach for finding multiple preset diagnosing experiments described
        by Arthur Gill in "Introduction to the Theory of Finite State Machines".
        Some optional parameterized adaptations, e.g., for computing suffix-closed sets target the application in
        L*-based learning and conformance testing.
        The function only works for minimal automata.

        :param list | None char_set_init: A list of sequence that will be included in the characterization set, e.g.,
            the input alphabet. An empty sequence is added to this list when using automata with state labels
            (DFA and Moore).
        :param bool online_suffix_closure: If true, ensures suffix closedness of the characterization set at every
            computation step.
        :param bool split_all_blocks: If false, the computation follows the original tree-based strategy, where
            newly computed sequences are only checked on a subset of the states to be distinguished. If true,
            sequences are used to distinguish all states, yielding a potentially smaller set, which is useful for
            conformance testing and learning.
        :param bool return_same_states: If True, a single distinguishable pair of states will be returned, or
            None, None if there are no non-distinguishable states.
        :param bool raise_warning: Prints warning message if characterization set cannot be computed.
        :return list | tuple | None: A characterization set, a pair of non-distinguishable states (if
            return_same_states is True), or None if a non-minimal automaton is passed to the function.
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

        unique_char_set = []
        for seq in char_set:
            if seq not in unique_char_set:
                unique_char_set.append(seq)
        char_set = unique_char_set

        if return_same_states:
            return None, None
        return char_set

    def _split_blocks(self, blocks: list, seq: tuple) -> list:
        """
        Refines a partition of states (blocks) using the output response to a given input sequence seq.

        :param list blocks: A partition of states.
        :param tuple seq: An input sequence.
        :return list: A refined partition of states.
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

    def compute_prefixes(self) -> None:
        """
        Computes and assigns the shortest access sequence (prefix) from the initial state for every state that
        does not already have one.
        """
        for s in self.states:
            if not s.prefix:
                s.prefix = self.get_shortest_path(self.initial_state, s)

    def minimize(self) -> None:
        """
        Minimizes the automaton in place by merging non-distinguishable states.
        """
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

    def __eq__(self, other: 'Automaton') -> bool:
        """
        :param Automaton other: Automaton to compare against.
        :return bool: True if this automaton and other are bisimilar, False otherwise.
        """
        from aalpy.utils import bisimilar
        return bisimilar(self, other)
