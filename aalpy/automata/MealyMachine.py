# Mealy machine state and automaton implementation, where outputs are associated with transitions.
from collections.abc import Hashable
from typing import Generic

from aalpy.base import AutomatonState, DeterministicAutomaton
from aalpy.base.Automaton import OutputType, InputType


class MealyState(AutomatonState, Generic[InputType, OutputType]):
    """
    Single state of a Mealy machine. Each state has an output_fun dictionary that maps inputs to outputs.
    """

    def __init__(self, state_id: Hashable) -> None:
        """
        Creates a Mealy machine state.

        :param Hashable state_id: Unique identifier of the state.
        """
        super().__init__(state_id)
        self.transitions: dict[InputType, MealyState[InputType, OutputType]] = dict()
        self.output_fun: dict[InputType, OutputType] = dict()


class MealyMachine(DeterministicAutomaton[MealyState[InputType, OutputType]]):
    """
    Deterministic Mealy machine, where outputs depend on the input and the current state.
    """

    def __init__(self, initial_state: MealyState, states: list[MealyState]) -> None:
        """
        Creates a Mealy machine.

        :param MealyState initial_state: Initial state of the Mealy machine.
        :param list[MealyState] states: All states of the Mealy machine.
        """
        super().__init__(initial_state, states)

    def step(self, letter: InputType) -> OutputType:
        """
        Performs a single step on the Mealy machine. In Mealy machines, outputs depend on the input and the
        current state.

        :param InputType letter: Single input that is looked up in the transition and output functions.
        :return OutputType: Output corresponding to the input from the current state.
        """
        output = self.current_state.output_fun[letter]
        self.current_state = self.current_state.transitions[letter]
        return output

    def to_state_setup(self) -> dict:
        """
        Converts the Mealy machine to a state setup dictionary.

        :return dict: Map from state_id to a transitions_dict mapping input to (output, target_state_id).
        """
        state_setup_dict = {}

        # ensure prefixes are computed
        self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix) if x.prefix is not None else len(self.states))
        for s in sorted_states:
            state_setup_dict[s.state_id] = {k: (s.output_fun[k], v.state_id) for k, v in s.transitions.items()}

        return state_setup_dict

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'MealyMachine':
        """
        Creates a Mealy machine from a state setup dictionary. The first state in the state setup is the initial
        state.

        Example state setup::

            state_setup = {
                "a": {"x": ("o1", "b1"), "y": ("o2", "a")},
                "b1": {"x": ("o3", "b2"), "y": ("o1", "a")},
                "b2": {"x": ("o1", "b3"), "y": ("o2", "a")},
                "b3": {"x": ("o3", "b4"), "y": ("o1", "a")},
                "b4": {"x": ("o1", "c"), "y": ("o4", "a")},
                "c": {"x": ("o3", "a"), "y": ("o5", "a")},
            }

        :param dict state_setup: Map from state_id to a transitions_dict mapping input to (output, target_state_id).
        :return MealyMachine: The constructed Mealy machine.
        """
        # state_setup should map from state_id to tuple(transitions_dict).
        # Each entry in transition dict is <input> : <output, new_state_id>

        # build states with state_id and output
        states = {key: MealyState(key) for key, _ in state_setup.items()}

        # add transitions to states
        for state_id, state in states.items():
            for _input, (output, new_state) in state_setup[state_id].items():
                state.transitions[_input] = states[new_state]
                state.output_fun[_input] = output

        # states to list
        states = [state for state in states.values()]

        # build moore machine with first state as starting state
        mm = MealyMachine(states[0], states)

        for state in states:
            state.prefix = mm.get_shortest_path(mm.initial_state, state)

        return mm
