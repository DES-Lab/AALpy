import random
from typing import List, Dict

from aalpy.base import AutomatonState, Automaton

class NDMooreState(AutomatonState):
    """
    Single state of a Moore machine. Each state has an output value.
    """

    def __init__(self, state_id, output=None):
        super().__init__(state_id)
        self.transitions : Dict[List] = dict()
        self.output = output


class NDMooreMachine(Automaton):

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
        options = self.current_state.transitions[letter]
        self.current_state = random.choice(options)
        return self.current_state.output
