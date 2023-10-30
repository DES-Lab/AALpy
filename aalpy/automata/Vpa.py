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
        self.target = target
        self.symbol = symbol
        self.action = action
        self.stack_guard = stack_guard

    def __str__(self):
        return f"{self.symbol}: {self.start.state_id} --> {self.target.state_id} | {self.action}: {self.stack_guard}"


class Vpa(Automaton):
    """
    Visibly Pushdown Automaton.
    """
    empty = "_"
    error_state = VpaState("ErrorSinkState", False)

    def __init__(self, initial_state: VpaState, states, input_alphabet: VpaAlphabet):
        super().__init__(initial_state, states)
        self.initial_state = initial_state
        self.states = states
        self.input_alphabet = input_alphabet
        self.current_state = None
        self.call_balance = 0
        self.stack = []

        # alphabet sets for faster inclusion checks (as in VpaAlphabet we have lists, for reproducibility)
        self.internal_set = set(self.input_alphabet.internal_alphabet)
        self.call_set = set(self.input_alphabet.call_alphabet)
        self.return_set = set(self.input_alphabet.return_alphabet)

    def reset_to_initial(self):
        super().reset_to_initial()
        self.reset()

    def reset(self):
        self.current_state = self.initial_state
        self.stack = [self.empty]
        self.call_balance = 0
        return self.current_state.is_accepting and self.top() == self.empty

    def top(self):
        return self.stack[-1]

    def pop(self):
        return self.stack.pop()

    def possible(self, letter):
        """
        Checks if a certain step on the automaton is possible
        """
        if self.current_state == Vpa.error_state:
            return True
        if letter is not None:
            transitions = self.current_state.transitions[letter]
            possible_trans = []
            for t in transitions:
                if t.symbol in self.call_set:
                    possible_trans.append(t)
                elif t.symbol in self.return_set:
                    if t.stack_guard == self.top():
                        possible_trans.append(t)
                elif t.symbol in self.internal_set:
                    possible_trans.append(t)
                else:
                    assert False and print(f'Letter {letter} is not part of any alphabet')
            assert len(possible_trans) < 2
            if len(possible_trans) == 0:
                return False
            else:
                return True
        return False

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
        if not self.possible(letter):
            self.current_state = Vpa.error_state
            return False
        if letter is not None:
            transitions = self.current_state.transitions[letter]
            possible_trans = []
            for t in transitions:
                if t.symbol in self.call_set:
                    possible_trans.append(t)
                elif t.symbol in self.return_set:
                    if t.stack_guard == self.top():
                        possible_trans.append(t)
                elif t.symbol in self.internal_set:
                    possible_trans.append(t)
                else:
                    assert False

            assert len(possible_trans) < 2
            trans = possible_trans[0]
            self.current_state = trans.target
            if trans.action == 'push':
                assert(letter in self.call_set)     # push letters must be in call set
                self.stack.append(trans.stack_guard)
            elif trans.action == 'pop':
                assert(letter in self.return_set)     # pop letters must be in return set
                if len(self.stack) <= 1:  # empty stack elem should always be there
                    self.current_state = Vpa.error_state
                    return False
                self.stack.pop()

        return self.current_state.is_accepting and self.top() == self.empty

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        # self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (
                s.is_accepting, {k: (v.target.state_id, v.action) for k, v in s.transitions.items()})

        return state_setup_dict

    @staticmethod
    def from_state_setup(state_setup: dict, init_state_id: str, input_alphabet: VpaAlphabet):
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

        vpa = Vpa(init_state, states, input_alphabet)
        return vpa
