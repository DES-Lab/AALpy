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

    def compute_characterization_set(self, char_set_init=None, online_suffix_closure=True, split_all_blocks=True,
                                     return_same_states=False, raise_warning=True):
        return super(Dfa, self).compute_characterization_set(char_set_init if char_set_init else [()],
                                                             online_suffix_closure, split_all_blocks,
                                                             return_same_states, raise_warning)

    def compute_output_seq(self, state, sequence):
        if not sequence:
            return [state.is_accepting]
        return super(Dfa, self).compute_output_seq(state, sequence)

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (s.is_accepting, {k: v.state_id for k, v in s.transitions.items()})

        return state_setup_dict

    def copy(self):
        from aalpy.utils import dfa_from_state_setup
        return dfa_from_state_setup(self.to_state_setup())