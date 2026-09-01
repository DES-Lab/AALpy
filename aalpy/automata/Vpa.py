# Visibly Pushdown Automaton (VPA) state and automaton implementation.
import random
from collections import defaultdict
from collections.abc import Hashable

from aalpy.automata import Dfa
from aalpy.base import Automaton, AutomatonState


class VpaAlphabet:
    """
    The Alphabet of a VPA.

    Attributes:
        internal_alphabet (list[str]): Letters for internal transitions.
        call_alphabet (list[str]): Letters for push transitions.
        return_alphabet (list[str]): Letters for pop transitions.
        exclusive_call_return_pairs (dict[str, str]): A dictionary representing exclusive pairs
            of call and return symbols.
    """

    def __init__(self, internal_alphabet: list[str], call_alphabet: list[str], return_alphabet: list[str],
                 exclusive_call_return_pairs: dict[str, str] | None = None) -> None:
        """
        Creates a VPA alphabet.

        :param list[str] internal_alphabet: Letters for internal transitions.
        :param list[str] call_alphabet: Letters for push transitions.
        :param list[str] return_alphabet: Letters for pop transitions.
        :param dict[str, str] | None exclusive_call_return_pairs: Exclusive pairs of call and return symbols.
        """
        self.internal_alphabet = internal_alphabet
        self.call_alphabet = call_alphabet
        self.return_alphabet = return_alphabet
        self.exclusive_call_return_pairs = exclusive_call_return_pairs

    def get_merged_alphabet(self) -> list[str]:
        """
        Get the merged alphabet, including internal, call, and return symbols.

        :return list[str]: A list of all symbols in the alphabet.
        """
        alphabet = list()
        alphabet.extend(self.internal_alphabet)
        alphabet.extend(self.call_alphabet)
        alphabet.extend(self.return_alphabet)
        return alphabet

    def __str__(self) -> str:
        """
        :return str: A string representation of the alphabet.
        """
        return f'Internal: {self.internal_alphabet} Call: {self.call_alphabet} Return: {self.return_alphabet}'


class VpaState(AutomatonState):
    """
    Single state of a VPA.
    """

    def __init__(self, state_id: Hashable, is_accepting: bool = False) -> None:
        """
        Creates a VPA state.

        :param Hashable state_id: Unique identifier of the state.
        :param bool is_accepting: Whether the state is an accepting state.
        """
        super().__init__(state_id)
        self.transitions: dict[str, list[VpaTransition]] = defaultdict(list)
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

    def __init__(self, start: VpaState, target: VpaState, symbol: str, action: str | None,
                 stack_guard: str | None = None) -> None:
        """
        Creates a VPA transition.

        :param VpaState start: The starting state of the transition.
        :param VpaState target: The target state of the transition.
        :param str symbol: The symbol associated with the transition.
        :param str | None action: The action performed during the transition (push | pop | None).
        :param str | None stack_guard: The stack symbol to be pushed/popped.
        """
        self.start = start
        self.target_state = target
        self.letter = symbol
        self.action = action
        self.stack_guard = stack_guard

    def __str__(self) -> str:
        """
        :return str: A string representation of the transition.
        """
        return f"{self.letter}: {self.start.state_id} --> {self.target_state.state_id} | {self.action}: {self.stack_guard}"


class Vpa(Automaton):
    """
    Visibly Pushdown Automaton.
    """
    error_state = VpaState("ErrorSinkState", False)

    def __init__(self, initial_state: VpaState, states: list[VpaState]) -> None:
        """
        Creates a VPA.

        :param VpaState initial_state: Initial state of the VPA.
        :param list[VpaState] states: All states of the VPA.
        """
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

    def reset_to_initial(self) -> None:
        """
        Resets the current state and stack of the VPA to the initial configuration.
        """
        self.current_state = self.initial_state
        self.stack = []

    def top(self) -> str | list:
        """
        :return str | list: The top of the stack, or an empty list if the stack is empty.
        """
        return self.stack[-1] if self.stack else []

    def step(self, letter: str | None) -> bool:
        """
        Perform a single step on the VPA by transitioning with the given input letter.

        :param str | None letter: A single input that is looked up in the transition table of the VpaState.
        :return bool: True if the reached state is an accepting state and the stack is empty, False otherwise.
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

    def execute_sequence(self, origin_state: VpaState, seq: list[str], stack: list) -> list[bool]:
        """
        Executes an input sequence on the VPA starting from a given state and stack configuration.

        A VPA's actual configuration is the pair (state, stack), not just the state, since the stack is
        what makes call/return symbols visibly balanced. The generic Automaton.execute_sequence only
        resets current_state, so it cannot start from a well-defined configuration on its own; stack is
        therefore a required parameter here rather than defaulted, so callers always state explicitly
        which configuration they mean to start from (pass [] to start fresh from origin_state).

        :param VpaState origin_state: State from which the sequence execution starts.
        :param list[str] seq: Input sequence to execute.
        :param list stack: Stack content to start from.
        :return list[bool]: The output response for the executed sequence.
        """
        self.current_state = origin_state
        self.stack = list(stack)
        return [self.step(s) for s in seq]

    def to_state_setup(self) -> dict:
        """
        Converts the VPA to a state setup dictionary.

        :return dict: Map from state_id to tuple(is_accepting, transitions_dict).
        """
        state_setup_dict = {}

        # ensure prefixes are computed
        # self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix) if x.prefix is not None else len(self.states))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (
                s.is_accepting,
                {k: [(t.target_state.state_id, t.action, t.stack_guard) for t in v] for k, v in s.transitions.items()})

        return state_setup_dict

    def get_input_alphabet(self) -> VpaAlphabet:
        """
        Computes the input alphabet of the VPA from its transitions.

        :return VpaAlphabet: The input alphabet.
        """
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
        Check whether all states have defined transition for all inputs.

        :return bool: True if automaton is input complete, False otherwise.
        """
        alphabet = set(self.get_input_alphabet().get_merged_alphabet())
        for state in self.states:
            if set(state.transitions.keys()) != alphabet:
                return False
        return True

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'Vpa':
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

        :param dict state_setup: A dictionary mapping from state IDs to tuples containing
            (is_accepting: bool, transitions_dict: dict), where transitions_dict maps input symbols to
            lists of tuples (target_state_id, action, stack_guard).
        :param init_state_id: The state ID for the initial state of the VPA, passed via kwargs.
        :return Vpa: The constructed Visibly Pushdown Automaton.
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

    def is_balanced(self, seq: list[str]) -> bool:
        """
        Checks whether an input sequence has balanced call and return symbols with respect to the VPA's alphabet.

        :param list[str] seq: The input sequence to check.
        :return bool: True if the sequence is balanced, False otherwise.
        """
        from aalpy.utils import is_balanced
        return is_balanced(seq, self.input_alphabet)

    def generate_random_accepting_word(self, min_steps: int = 4, max_steps: int = 20) -> list[str] | None:
        """
        Generate a random valid sequence for a given VPDA.

        :param int min_steps: Minimum number of steps.
        :param int max_steps: Maximum number of steps before the process terminates.
        :return list[str] | None: A list of input symbols (the generated sequence) leading to an accepting state,
            or None if a sequence could not be generated.
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


def vpa_from_dfa_representation(dfa_repr: Dfa, vpa_alphabet: VpaAlphabet) -> Vpa:
    """
    Converts a DFA representation of a VPA (where call/return symbols may be encoded as tuples with the top of
    stack) into an equivalent Vpa.

    :param Dfa dfa_repr: The DFA representation to convert.
    :param VpaAlphabet vpa_alphabet: The alphabet of the resulting VPA.
    :return Vpa: The constructed VPA.
    """
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
