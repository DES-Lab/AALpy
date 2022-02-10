from aalpy.base import AutomatonState, DeterministicAutomaton


class MooreState(AutomatonState):
    """
    Single state of a Moore machine. Each state has an output value.
    """

    def __init__(self, state_id, output):
        super().__init__(state_id)
        self.output = output


class MooreMachine(DeterministicAutomaton):

    def __init__(self, initial_state: AutomatonState, states: list):
        super().__init__(initial_state, states)

    def step(self, letter):
        """
        In Moore machines outputs depend on the current state.

        Args:

            letter: single input that is looked up in the transition function leading to a new state

        Returns:

            the output of the reached state

        """
        if letter is not None:
            self.current_state = self.current_state.transitions[letter]
        return self.current_state.output

    def compute_characterization_set(self, char_set_init=None, online_suffix_closure=True, split_all_blocks=True):
        return super(MooreMachine, self).compute_characterization_set(char_set_init if char_set_init else [()],
                                                             online_suffix_closure,
                                                             split_all_blocks)