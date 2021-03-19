from aalpy.base import Automaton, AutomatonState


class MooreState(AutomatonState):
    """
    Single state of a Moore machine. Each state has an output value.
    """

    def __init__(self, state_id, output):
        super().__init__(state_id)
        self.output = output


class MooreMachine(Automaton):

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
        self.current_state = self.current_state.transitions[letter]
        return self.current_state.output
