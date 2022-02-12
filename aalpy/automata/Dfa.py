from aalpy.base import AutomatonState, DeterministicAutomaton


class DfaState(AutomatonState):
    """
    Single state of a deterministic finite automaton.
    """

    def __init__(self, state_id, is_accepting=False):
        super().__init__(state_id)
        self.is_accepting = is_accepting


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
