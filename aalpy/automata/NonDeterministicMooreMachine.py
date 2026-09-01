# Non-deterministic Moore machine state and automaton implementation.
import random
from collections import defaultdict
from collections.abc import Hashable
from typing import Generic

from aalpy.base import AutomatonState, Automaton
from aalpy.base.Automaton import OutputType, InputType


class NDMooreState(AutomatonState, Generic[InputType, OutputType]):
    """
    Single state of a non-deterministic Moore machine. Each state has an output value.
    """

    def __init__(self, state_id: Hashable, output: OutputType | None = None) -> None:
        """
        Creates a non-deterministic Moore machine state.

        :param Hashable state_id: Unique identifier of the state.
        :param OutputType | None output: Output value associated with the state.
        """
        super().__init__(state_id)
        self.transitions: dict[InputType, list[NDMooreState[InputType, OutputType]]] = defaultdict(lambda: list())
        self.output: OutputType | None = output


class NDMooreMachine(Automaton[NDMooreState[InputType, OutputType]]):
    """
    Non-deterministic Moore machine, where outputs depend on the current state and transitions are chosen
    non-deterministically.
    """

    def to_state_setup(self) -> dict:
        """
        Converts the non-deterministic Moore machine to a state setup dictionary.

        :return dict: Map from state_id to tuple(output, transitions_dict).
        """
        state_setup = dict()

        def set_dict_entry(state: NDMooreState) -> None:
            """
            Adds a single state's configuration to the enclosing state_setup dictionary.

            :param NDMooreState state: State to add.
            """
            state_setup[state.state_id] = (state.output,
                                           {in_sym: [target.state_id for target in trans] for in_sym, trans in
                                            state.transitions.items()})

        set_dict_entry(self.initial_state)
        for state in self.states:
            if state is self.initial_state:
                continue
            set_dict_entry(state)

        return state_setup

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'NDMooreMachine':
        """
        Creates a non-deterministic Moore machine from a state setup dictionary. The first state in the state
        setup is the initial state.

        :param dict state_setup: Map from state_id to tuple(output, transitions_dict).
        :return NDMooreMachine: The constructed non-deterministic Moore machine.
        """
        states_map = {key: NDMooreState(key, output=value[0]) for key, value in state_setup.items()}

        for key, values in state_setup.items():
            source = states_map[key]
            for i, transitions in values[1].items():
                for node in transitions:
                    source.transitions[i].append(states_map[node])

        initial_state = states_map[list(state_setup.keys())[0]]
        return NDMooreMachine(initial_state, list(states_map.values()))

    def __init__(self, initial_state: AutomatonState, states: list) -> None:
        """
        Creates a non-deterministic Moore machine.

        :param AutomatonState initial_state: Initial state of the non-deterministic Moore machine.
        :param list states: All states of the non-deterministic Moore machine.
        """
        super().__init__(initial_state, states)

    def step(self, letter: InputType) -> OutputType:
        """
        In Moore machines outputs depend on the current state.

        :param InputType letter: Single input that is looked up in the transition function leading to a new state.
        :return OutputType: The output of the reached state.
        """
        options = self.current_state.transitions[letter]
        self.current_state = random.choice(options)
        return self.current_state.output
