from aalpy.base import AutomatonState, DeterministicAutomaton


class DfaState(AutomatonState):
    """
    Single state of a deterministic finite automaton.
    """

    def __init__(self, state_id, is_accepting=False):
        super().__init__(state_id)
        self.is_accepting = is_accepting

    def __repr__(self):
        id = str(self.state_id) if self.state_id != (None,) else '(None)'
        return f"{self.__class__.__name__}'{id}' " + ("(accepting)" if self.is_accepting else "")


class Dfa(DeterministicAutomaton):
    """
    Deterministic finite automaton.
    """

    def __init__(self, initial_state: DfaState, states):
        super().__init__(initial_state, states)

    def step(self, letter):
        """
        Args:

            letter: single input that is looked up in the transition table of the DfaState

        Returns:

            True if the reached state is an accepting state, False otherwise
        """
        if letter is not None:
            self.current_state = self.current_state.transitions[letter]
        return self.current_state.is_accepting

    def compute_characterization_set(self, char_set_init=None, online_suffix_closure=True, split_all_blocks=True):
        return super(Dfa, self).compute_characterization_set(char_set_init if char_set_init else [()],
                                                             online_suffix_closure,
                                                             split_all_blocks)

    def is_minimal(self):
        return self.compute_characterization_set() != []

    def get_result(self, input: tuple):
        """
        Args:
            input: query

        Returns:
            the final result of the query <input>, starting from the initial state
        """
        saved_state = self.current_state
        res = self.execute_sequence(self.initial_state, input)[-1]
        self.current_state = saved_state
        return res
