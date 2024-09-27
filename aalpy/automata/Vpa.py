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
                    if transition.action == 'push':
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

    def gen_random_accepting_word(self, return_letter_prob: float = 0.5, call_letter_prob: float = 0.5,
                                  early_finish: bool = True):
        """
        Create a random word that gets accepted by the automaton.

        Args:

        Returns:
        """
        assert return_letter_prob + call_letter_prob <= 1.0

        internal_letter_prob = 1 - call_letter_prob - return_letter_prob

        assert (call_letter_prob + return_letter_prob + internal_letter_prob) == 1.0

        call_letter_boarder = call_letter_prob
        return_letter_boarder = call_letter_boarder + return_letter_prob
        internal_letter_boarder = return_letter_boarder + internal_letter_prob

        word = []
        self.reset_to_initial()
        while True:
            letter_type = random.uniform(0.0, 1.0)
            if 0.0 <= letter_type <= call_letter_boarder:
                possible_letters = self.input_alphabet.call_alphabet
            elif call_letter_boarder < letter_type <= return_letter_boarder:
                # skip return letters if stack is empty or if the word is empty
                if not self.stack:
                    continue
                possible_letters = self.input_alphabet.return_alphabet
            elif return_letter_boarder < letter_type <= internal_letter_boarder:
                possible_letters = self.input_alphabet.internal_alphabet
            else:
                assert False

            assert len(possible_letters) > 0

            letter = ''
            if early_finish:
                for l in possible_letters:
                    for transition in self.current_state.transitions[l]:
                        if transition.target_state.is_accepting:
                            letter = l
                            break
                    break
            if letter == '':
                random_trans_letter_index = random.randint(0, len(possible_letters) - 1)
                letter = possible_letters[random_trans_letter_index]
            self.step(letter)
            if not self.current_state == self.error_state:
                word.append(letter)
            else:
                self.reset_to_initial()
                self.execute_sequence(self.initial_state, word)

            if self.current_state.is_accepting and not self.stack:
                break

        return word


def from_dfa_representation(dfa_repr, vpa_alphabet):
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



