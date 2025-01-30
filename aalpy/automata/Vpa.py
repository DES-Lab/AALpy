import random
from collections import defaultdict
from typing import List, Dict

from aalpy.base import Automaton, AutomatonState


class VpaAlphabet:
    """
    The Alphabet of a VPA.

    Attributes:
        internal_alphabet (List[str]): Letters for internal transitions.
        call_alphabet (List[str]): Letters for push transitions.
        return_alphabet (List[str]): Letters for pop transitions.
        exclusive_call_return_pairs (Dict[str, str]): A dictionary representing exclusive pairs
            of call and return symbols.
    """

    def __init__(self, internal_alphabet: List[str], call_alphabet: List[str], return_alphabet: List[str],
                 exclusive_call_return_pairs: Dict[str, str] = None):
        self.internal_alphabet = internal_alphabet
        self.call_alphabet = call_alphabet
        self.return_alphabet = return_alphabet
        self.exclusive_call_return_pairs = exclusive_call_return_pairs

    def get_merged_alphabet(self) -> List[str]:
        """
        Get the merged alphabet, including internal, call, and return symbols.

        Returns:
            List[str]: A list of all symbols in the alphabet.
        """
        alphabet = list()
        alphabet.extend(self.internal_alphabet)
        alphabet.extend(self.call_alphabet)
        alphabet.extend(self.return_alphabet)
        return alphabet

    def __str__(self) -> str:
        """
        Returns:
            str: A string representation of the alphabet.
        """
        return f'Internal: {self.internal_alphabet} Call: {self.call_alphabet} Return: {self.return_alphabet}'


class VpaState(AutomatonState):
    """
    Single state of a VPA.
    """

    def __init__(self, state_id, is_accepting=False):
        super().__init__(state_id)
        self.transitions = defaultdict(list)
        self.is_accepting = is_accepting


class VpaTransition:
    """
    Represents a transition in a VPA.

    Attributes:
        start (VpaState): The starting state of the transition.
        target (VpaState): The target state of the transition.
        symbol: The symbol associated with the transition.
        action: The action performed during the transition (push | pop | None).
        stack_guard: The stack symbol to be pushed/popped.
    """

    def __init__(self, start: VpaState, target: VpaState, symbol, action, stack_guard=None):
        self.start = start
        self.target_state = target
        self.letter = symbol
        self.action = action
        self.stack_guard = stack_guard

    def __str__(self):
        return f"{self.letter}: {self.start.state_id} --> {self.target_state.state_id} | {self.action}: {self.stack_guard}"


class Vpa(Automaton):
    """
    Visibly Pushdown Automaton.
    """
    error_state = VpaState("ErrorSinkState", False)

    def __init__(self, initial_state: VpaState, states):
        super().__init__(initial_state, states)
        self.initial_state = initial_state
        self.states = states
        self.input_alphabet = self.get_input_alphabet()
        self.current_state = None
        self.stack = []

        # alphabet sets for faster inclusion checks (as in VpaAlphabet we have lists, for reproducibility)
        self.internal_set = set(self.input_alphabet.internal_alphabet)
        self.call_set = set(self.input_alphabet.call_alphabet)
        self.return_set = set(self.input_alphabet.return_alphabet)

    def reset_to_initial(self):
        self.current_state = self.initial_state
        self.stack = []

    def top(self):
        return self.stack[-1] if self.stack else []

    def step(self, letter):
        """
        Perform a single step on the VPA by transitioning with the given input letter.

        Args:
            letter: A single input that is looked up in the transition table of the VpaState.

        Returns:
            bool: True if the reached state is an accepting state and the stack is empty, False otherwise.
        """
        if self.current_state == Vpa.error_state:
            return False

        if letter is None:
            return self.current_state.is_accepting and self.stack == []

        transitions = self.current_state.transitions[letter]

        taken_transition = None

        for t in transitions:
            if t.action == 'push' or t.action is None:
                taken_transition = t
                break
            else:
                if t.stack_guard == self.top():
                    taken_transition = t
                    break

        if taken_transition is None:
            self.current_state = Vpa.error_state
            return False

        self.current_state = taken_transition.target_state
        if taken_transition.action == 'push':
            self.stack.append(taken_transition.stack_guard)
        elif taken_transition.action == 'pop':
            # empty stack elem should always be there
            if not self.stack:
                self.current_state = Vpa.error_state
                return False
            self.stack.pop()

        return self.current_state.is_accepting and self.stack == []

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        # self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (
                s.is_accepting, {k: (v.target_state.state_id, v.action) for k, v in s.transitions.items()})

        return state_setup_dict

    def get_input_alphabet(self) -> VpaAlphabet:
        int_alphabet, ret_alphabet, call_alphabet = [], [], []
        for state in self.states:
            for transition_list in state.transitions.values():
                for transition in transition_list:
                    if transition.action == 'pop':
                        if transition.letter not in ret_alphabet:
                            ret_alphabet.append(transition.letter)
                    elif transition.action == 'push':
                        if transition.letter not in call_alphabet:
                            call_alphabet.append(transition.letter)
                    elif transition.letter not in int_alphabet:
                        int_alphabet.append(transition.letter)

        return VpaAlphabet(int_alphabet, call_alphabet, ret_alphabet)

    def is_input_complete(self) -> bool:
        """
        Check whether all states have defined transition for all inputs
        :return: true if automaton is input complete

        Returns:

            True if input complete, False otherwise

        """
        alphabet = set(self.get_input_alphabet().get_merged_alphabet())
        for state in self.states:
            if set(state.transitions.keys()) != alphabet:
                return False
        return True

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs):
        """
        Create a VPA from a state setup.

        Example state setup:
            state_setup = {
                "q0": (False, {"(": [("q1", 'push', "(")],
                               "[": [("q1", 'push', "[")],  # exclude empty seq
                               }),
                "q1": (False, {"(": [("q1", 'push', "(")],
                               "[": [("q1", 'push', "[")],
                               ")": [("q2", 'pop', "(")],
                               "]": [("q2", 'pop', "[")]}),
                "q2": (True, {
                    ")": [("q2", 'pop', "(")],
                    "]": [("q2", 'pop', "[")]
                }),

            Args:
                state_setup (dict): A dictionary mapping from state IDs to tuples containing
                    (is_accepting: bool, transitions_dict: dict), where transitions_dict maps input symbols to
                    lists of tuples (target_state_id, action, stack_guard).
                init_state_id (str): The state ID for the initial state of the VPA.
                input_alphabet (VpaAlphabet): The alphabet for the VPA.

            Returns:
                Vpa: The constructed Variable Pushdown Automaton.
            """
        # state_setup should map from state_id to tuple(is_accepting and transitions_dict)

        init_state_id = kwargs['init_state_id']

        # build states with state_id and output
        states = {key: VpaState(key, val[0]) for key, val in state_setup.items()}
        states[Vpa.error_state.state_id] = Vpa.error_state  # PdaState(Pda.error_state,False)
        # add transitions to states
        for state_id, state in states.items():
            if state_id == Vpa.error_state.state_id:
                continue
            for _input, trans_spec in state_setup[state_id][1].items():
                for (target_state_id, action, stack_guard) in trans_spec:
                    trans = VpaTransition(start=state, target=states[target_state_id], symbol=_input, action=action,
                                          stack_guard=stack_guard)
                    state.transitions[_input].append(trans)

        init_state = states[init_state_id]
        # states to list
        states = [state for state in states.values()]

        vpa = Vpa(init_state, states)
        return vpa

    def is_balanced(self, seq):
        from aalpy.utils import is_balanced
        return is_balanced(seq, self.input_alphabet)

    def generate_random_accepting_word(self, min_steps=4, max_steps=20):
        """
        Generate a random valid sequence for a given VPDA.

        Args:

            min_steps : Minimum number of steps
            max_steps : Maximum number of steps before the process terminates

        Returns:
            list: A list of input symbols (the generated sequence) leading to an accepting state.
        """

        sequence = []
        self.reset_to_initial()

        for step_count in range(max_steps):
            current_state = self.current_state

            # If we have met the min_steps requirement and are in an accepting state with an empty stack, stop
            if step_count >= min_steps and current_state.is_accepting and not self.stack:
                return sequence

            # Get all possible transitions from the current state
            possible_transitions = []
            for letter, transitions in current_state.transitions.items():
                for t in transitions:
                    if t.action == 'pop' and self.stack and t.stack_guard == self.top():
                        possible_transitions.append(t)
                    elif t.action == 'push' or t.action is None:
                        possible_transitions.append(t)

            # If no valid transitions exist, return an incomplete sequence or error
            if not possible_transitions:
                break

            # Randomly choose a valid transition
            chosen_transition = random.choice(possible_transitions)

            # Perform the transition
            self.step(chosen_transition.letter)

            # Add the chosen letter to the sequence
            sequence.append(chosen_transition.letter)

        # None indicates that a sequance was not successfully generated
        return None

    def output_step(self, state, letter):
        state_save = self.current_state
        stack_save = self.stack.copy()
        self.current_state = state
        output = self.step(letter)
        stack = self.stack
        self.current_state = state_save
        self.stack = stack_save
        return output, stack

    def find_distinguishing_seq(self, state1, state2, alphabet):

        visited = set()
        to_explore = [(state1, state2, [])]
        while to_explore:
            (curr_s1, curr_s2, prefix) = to_explore.pop(0)
            visited.add((curr_s1, curr_s2))
            for i in alphabet:
                o1, stack = self.output_step(curr_s1, i)
                o2, stack = self.output_step(curr_s2, i)
                new_prefix = prefix + [i]
                if o1 != o2:
                    return new_prefix
                else:
                    if curr_s1.transitions[i]:
                        next_s1 = curr_s1.transitions[i][0].target_state
                    else:
                        next_s1 = curr_s1
                    if curr_s2.transitions[i]:
                        next_s2 = curr_s2.transitions[i][0].target_state
                    else:
                        next_s2 = curr_s2

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
        stack_save = self.stack.copy()
        output = self.execute_sequence(state, sequence)
        self.current_state = state_save
        self.stack = stack_save
        return output

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
        import copy

        blocks = list()
        blocks.append(copy.copy(self.states))
        char_set = [] if not char_set_init else char_set_init
        if char_set_init:
            for seq in char_set_init:
                blocks = self._split_blocks(blocks, seq)

        alphabet = self.get_input_alphabet().get_merged_alphabet()
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


def vpa_from_dfa_representation(dfa_repr, vpa_alphabet):
    vpa_states = dict()
    for dfa_state in dfa_repr.states:
        vpa_state = VpaState(state_id=dfa_state.state_id, is_accepting=dfa_state.is_accepting)
        vpa_states[dfa_state.state_id] = vpa_state

    for dfa_state in dfa_repr.states:

        for input_symbol, reached_dfa_state in dfa_state.transitions.items():
            origin_state = vpa_states[dfa_state.state_id]
            reached_state = vpa_states[reached_dfa_state.state_id]

            top_of_stack = None
            if isinstance(input_symbol, tuple):
                input_symbol, top_of_stack = input_symbol[0], input_symbol[1]

            if input_symbol in vpa_alphabet.return_alphabet:
                transition = VpaTransition(origin_state, reached_state, input_symbol, 'pop', top_of_stack)
            else:
                action = 'push' if input_symbol in vpa_alphabet.call_alphabet else None
                stack_guard = input_symbol if action == 'push' else None
                transition = VpaTransition(origin_state, reached_state, input_symbol,
                                           action, stack_guard)

            origin_state.transitions[input_symbol].append(transition)

    initial_state = vpa_states[dfa_repr.initial_state.state_id]
    learned_model = Vpa(initial_state, list(vpa_states.values()))

    return learned_model
