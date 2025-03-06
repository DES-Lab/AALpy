import random
from collections import defaultdict
from typing import List, Dict, Generic

from aalpy.base import AutomatonState, Automaton
from aalpy.base.Automaton import OutputType, InputType


class NDMooreState(AutomatonState, Generic[InputType, OutputType]):
    """
    Single state of a non-deterministic Moore machine. Each state has an output value.
    """

    def __init__(self, state_id, output=None):
        super().__init__(state_id)
        self.transitions: Dict[InputType, List['NDMooreState']] = defaultdict(lambda: list())
        self.output: OutputType = output


class NDMooreMachine(Automaton[NDMooreState[InputType, OutputType]]):

    def to_state_setup(self):
        state_setup = dict()

        def set_dict_entry(state: NDMooreState):
            state_setup[state.state_id] = (state.output,
                                           {in_sym: [target.state_id for target in trans] for in_sym, trans in
                                            state.transitions.items()})

        set_dict_entry(self.initial_state)
        for state in self.states:
            if state is self.initial_state:
                continue
            set_dict_entry(state)

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'NDMooreMachine':
        states_map = {key: NDMooreState(key, output=value[0]) for key, value in state_setup.items()}

        for key, values in state_setup.items():
            source = states_map[key]
            for i, transitions in values[1].items():
                for node in transitions:
                    source.transitions[i].append(states_map[node])

        initial_state = states_map[list(state_setup.keys())[0]]
        return NDMooreMachine(initial_state, list(states_map.values()))

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
