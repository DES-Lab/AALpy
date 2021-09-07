import random
from collections import defaultdict

from aalpy.base import Automaton, AutomatonState


class McState(AutomatonState):
    def __init__(self, state_id, output):
        super().__init__(state_id)
        self.output = output
        # each child is a tuple (Node(output), probability)
        self.transitions = defaultdict(list)


class MarkovChain(Automaton):
    """Markov Decision Process."""

    def __init__(self, initial_state, states: list):
        super().__init__(initial_state, states)

    def reset_to_initial(self):
        self.current_state = self.initial_state

    def step(self, letter=None):
        """Next step is determined based on transition probabilities of the current state.

        Args:

            letter: input

        Returns:

            output of the current state
        """

        prob = random.random()

        probability_distributions = [j[1] for i in self.current_state.transitions.values() for j in i]
        states = [j[0] for i in self.current_state.transitions.values() for j in i]

        if not states:
            return self.current_state.output

        index = 0
        for i, p in enumerate(probability_distributions):
            prob -= p
            if prob <= 0:
                index = i
                break

        self.current_state = states[index]
        return self.current_state.output

    def step_to(self, input):
        """Performs a step on the automaton based on the input `inp` and output `out`.

        Args:

            input: input

        Returns:

            output of the reached state, None otherwise
        """
        if input in self.current_state.transitions.keys():
            self.current_state = self.current_state.transitions[input]
            return self.current_state.output
        return None
