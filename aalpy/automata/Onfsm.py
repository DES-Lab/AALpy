from collections import defaultdict
from random import choice

from aalpy.base import Automaton, AutomatonState


class OnfsmState(AutomatonState):
    """ """
    def __init__(self, state_id):
        super().__init__(state_id)
        # key/input maps to the list of tuples of possible output/new state [(output1, state1), (output2, state2)]
        self.transitions = defaultdict(list)

    def add_transition(self, inp, out, new_state):
        """

        Args:
          inp: 
          out: 
          new_state: 

        Returns:

        """
        self.transitions[inp].append((out, new_state))

    def get_transition(self, input, output=None):
        """

        Args:
          input: 
          output:  (Default value = None)

        Returns:

        """
        possible_transitions = self.transitions[input]
        if output:
            return next((t for t in possible_transitions if t[0] == output), None)
        else:
            return possible_transitions


class Onfsm(Automaton):
    """
    Observable non-deterministic finite state automaton.
    """
    def __init__(self, initial_state: OnfsmState, states: list):
        super().__init__(initial_state, states)

    def step(self, letter):
        """Next step is determined based on a uniform distribution over all transitions with the input 'letter'.

        Args:

            letter: input

        Returns:

            output of the probabilistically chosen transition

        """
        transition = choice(self.current_state.transitions[letter])
        output = transition[0]
        self.current_state = transition[1]
        return output

    def outputs_on_input(self, letter):
        """All possible observable outputs after executing the current input 'letter'.

        Args:

            letter: input

        Returns:

            list of observeable outputs

        """
        return [trans[0] for trans in self.current_state.transitions[letter]]

    def step_to(self, inp, out):
        """Performs a step on the automaton based on the input `inp` and output `out`.

        Args:

            inp: input
            out: output

        Returns:

            output of the reached state, None otherwise

        """
        for new_state in self.current_state.transitions[inp]:
            if new_state[0] == out:
                self.current_state = new_state[1]
                return out
        return None
