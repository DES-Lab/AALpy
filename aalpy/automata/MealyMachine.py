from aalpy.base import Automaton, AutomatonState


class MealyState(AutomatonState):
    """
    Single state of a Mealy machine. Each state has an output_fun dictionary that maps inputs to outputs.
    """
    def __init__(self, state_id):
        super().__init__(state_id)
        self.output_fun = dict()


class MealyMachine(Automaton):

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
