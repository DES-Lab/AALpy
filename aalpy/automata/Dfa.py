from aalpy.base import AutomatonState, Automaton


class DfaState(AutomatonState):
    """
    Single state of a deterministic finite automaton.
    """
    def __init__(self, state_id, is_accepting: bool = False):
        super().__init__(state_id)
        self.is_accepting = is_accepting


class Dfa(Automaton):
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
        self.current_state = self.current_state.transitions[letter]
        return self.current_state.is_accepting
