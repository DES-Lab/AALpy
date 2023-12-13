import random
from collections import defaultdict
from typing import Dict, Generic, List, Tuple

from aalpy.base import Automaton, AutomatonState
from aalpy.base.Automaton import OutputType, InputType


class MdpState(AutomatonState, Generic[InputType, OutputType]):
    """
    For transitions, each transition is a tuple (Node(output), probability)
    """
    def __init__(self, state_id, output=None):
        super().__init__(state_id)
        self.output: OutputType = output
        # each transition is a tuple (Node(output), probability)
        self.transitions: Dict[InputType, List[Tuple[MdpState, float]]] = defaultdict(list)


class Mdp(Automaton[MdpState[InputType, OutputType]]):
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

        probability_distributions = [i[1] for i in self.current_state.transitions[letter]]
        states = [i[0] for i in self.current_state.transitions[letter]]

        new_state = random.choices(states, probability_distributions, k=1)[0]

        self.current_state = new_state
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

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure initial state is first in the list
        if self.states[0] != self.initial_state:
            self.states.remove(self.initial_state)
            self.states.insert(0, self.initial_state)

        for s in self.states:
            state_setup_dict[s.state_id] = (s.output, {k: [(node.state_id, prob) for node, prob in v]
                                                       for k, v in s.transitions.items()})

        return state_setup_dict

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs):
        states_map = {key: MdpState(key, output=value[0]) for key, value in state_setup.items()}

        for key, values in state_setup.items():
            source = states_map[key]
            for i, transitions in values[1].items():
                for node, prob in transitions:
                    source.transitions[i].append((states_map[node], prob))

        initial_state = states_map[list(state_setup.keys())[0]]
        return Mdp(initial_state, list(states_map.values()))
