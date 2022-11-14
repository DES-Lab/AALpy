from aalpy.base import AutomatonState, DeterministicAutomaton


class MealyState(AutomatonState):
    """
    Single state of a Mealy machine. Each state has an output_fun dictionary that maps inputs to outputs.
    """

    def __init__(self, state_id):
        super().__init__(state_id)
        self.output_fun = dict()


class MealyMachine(DeterministicAutomaton):

    def __init__(self, initial_state: MealyState, states):
        super().__init__(initial_state, states)

    def step(self, letter):
        """
        In Mealy machines, outputs depend on the input and the current state.

            Args:

                letter: single input that is looked up in the transition and output functions

            Returns:

                output corresponding to the input from the current state
        """
        output = self.current_state.output_fun[letter]
        self.current_state = self.current_state.transitions[letter]
        return output

    def is_minimal(self):
        return self.compute_characterization_set(raise_warning=False) is not None

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix))
        for s in sorted_states:
            state_setup_dict[s.state_id] = {k: (s.output_fun[k], v.state_id) for k, v in s.transitions.items()}

        return state_setup_dict
