import os
import random
from collections import defaultdict

from aalpy.base import Automaton, AutomatonState


class MdpState(AutomatonState):
    def __init__(self, state_id, output=None):
        super().__init__(state_id)
        self.output = output
        # each child is a tuple (Node(output), probability)
        self.transitions = defaultdict(list)


class Mdp(Automaton):
    """Markov Decision Process."""
    def __init__(self, initial_state: MdpState, states: list):
        super().__init__(initial_state, states)

    def reset_to_initial(self):
        self.current_state = self.initial_state

    def step(self, letter):
        """Next step is determined based on transition probabilities of the current state.

        Args:

            letter: input

        Returns:

            output of the current state
        """
        if letter is None:
            return self.current_state.output
        prob = random.random()

        probability_distributions = [i[1] for i in self.current_state.transitions[letter]]
        index = 0
        for i, p in enumerate(probability_distributions):
            prob -= p
            if prob <= 0:
                index = i
                break

        self.current_state = self.current_state.transitions[letter][index][0]
        return self.current_state.output

    def step_to(self, inp, out):
        """Performs a step on the automaton based on the input `inp` and output `out`.

        Args:

            inp: input
            out: output

        Returns:

            output of the reached state, None otherwise
        """
        for new_state in self.current_state.transitions[inp]:
            if new_state[0].output == out:
                self.current_state = new_state[0]
                return out
        return None
