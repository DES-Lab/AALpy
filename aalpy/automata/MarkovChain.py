import random
from typing import Generic, Tuple, List

from aalpy.base import Automaton, AutomatonState
from aalpy.base.Automaton import OutputType


class McState(AutomatonState, Generic[OutputType]):
    def __init__(self, state_id, output):
        super().__init__(state_id)
        self.output: OutputType = output
        # transitions is a list of tuples (Node(output), probability)
        self.transitions: List[Tuple[McState, float]] = list()


class MarkovChain(Automaton[McState[OutputType]]):
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

        if not self.current_state.transitions:
            return self.current_state.output

        probability_distributions = [i[1] for i in self.current_state.transitions]
        states = [i[0] for i in self.current_state.transitions]

        new_state = random.choices(states, probability_distributions, k=1)[0]

        self.current_state = new_state
        return self.current_state.output

    def step_to(self, input):
        """Performs a step on the automaton based on the input `inp` and output `out`.

        Args:

            input: input

        Returns:

            output of the reached state, None otherwise
        """
        for s in self.current_state.transitions:
            if s[0].output == input:
                self.current_state = s[0]
                return self.current_state.output
        return None

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs):
        raise NotImplementedError()  # TODO implement

    def to_state_setup(self):
        raise NotImplementedError()  # TODO implement
