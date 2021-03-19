import random
from collections import defaultdict

from aalpy.base import Automaton, AutomatonState


class StochasticMealyState(AutomatonState):
    """ """
    def __init__(self, state_id):
        super().__init__(state_id)
        # each child is a tuple (newNode, output, probability)
        self.transitions = defaultdict(list)


class StochasticMealyMachine(Automaton):

    def __init__(self, initial_state: StochasticMealyState, states: list):
        super().__init__(initial_state, states)

    def reset_to_initial(self):
        self.current_state = self.initial_state

    def step(self, letter):
        """
        Next step is determined based on transition probabilities of the current state.

        Args:

           letter: input

        Returns:

           output of the current state
        """
        prob = random.random()
        probability_distributions = [i[2] for i in self.current_state.transitions[letter]]
        index = 0
        for i, p in enumerate(probability_distributions):
            prob -= p
            if prob <= 0:
                index = i
                break

        transition = self.current_state.transitions[letter][index]
        self.current_state = transition[0]
        return transition[1]

    def step_to(self, inp, out):
        """Performs a step on the automaton based on the input `inp` and output `out`.

        Args:

            inp: input
            out: output

        Returns:

            output of the reached state, None otherwise

        """
        for (new_state, output, prob) in self.current_state.transitions[inp]:
            if output == out:
                self.current_state = new_state
                return out
        return None
