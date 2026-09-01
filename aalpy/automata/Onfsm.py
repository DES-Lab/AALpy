# Observable non-deterministic finite state automaton (ONFSM) state and automaton implementation.
from collections import defaultdict
from collections.abc import Hashable
from random import choice
from typing import Generic

from aalpy.base import Automaton, AutomatonState
from aalpy.base.Automaton import OutputType, InputType


class OnfsmState(AutomatonState, Generic[InputType, OutputType]):
    """
    Single state of an observable non-deterministic finite state automaton.
    """

    def __init__(self, state_id: Hashable) -> None:
        """
        Creates an ONFSM state.

        :param Hashable state_id: Unique identifier of the state.
        """
        super().__init__(state_id)
        # TODO this order is inconsistent with probabilistic models
        # key/input maps to the list of tuples of possible output/new state [(output1, state1), (output2, state2)]
        self.transitions: dict[InputType, list[tuple[OutputType, OnfsmState[InputType, OutputType]]]] = defaultdict(list)

    def add_transition(self, inp: InputType, out: OutputType, new_state: 'OnfsmState[InputType, OutputType]') -> None:
        """
        Adds a transition from this state.

        :param InputType inp: Input triggering the transition.
        :param OutputType out: Output produced by the transition.
        :param OnfsmState new_state: Target state of the transition.
        """
        self.transitions[inp].append((out, new_state))

    def get_transition(self, input: InputType, output: OutputType | None = None) \
            -> list[tuple[OutputType, 'OnfsmState[InputType, OutputType]']] | tuple[OutputType, 'OnfsmState[InputType, OutputType]'] | None:
        """
        Looks up the possible transitions for a given input, optionally filtered by output.

        :param InputType input: Input to look up.
        :param OutputType | None output: If given, only the transition matching this output is returned.
        :return list[tuple[OutputType, OnfsmState]] | tuple[OutputType, OnfsmState] | None: All possible
            transitions for the input, or the single matching transition if output is given, or None if not found.
        """
        possible_transitions = self.transitions[input]
        if output:
            return next((t for t in possible_transitions if t[0] == output), None)
        else:
            return possible_transitions


class Onfsm(Automaton[OnfsmState[InputType, OutputType]]):
    """
    Observable non-deterministic finite state automaton.
    """

    def __init__(self, initial_state: OnfsmState, states: list) -> None:
        """
        Creates an ONFSM.

        :param OnfsmState initial_state: Initial state of the ONFSM.
        :param list states: All states of the ONFSM.
        """
        super().__init__(initial_state, states)

    def step(self, letter: InputType) -> OutputType:
        """Next step is determined based on a uniform distribution over all transitions with the input 'letter'.

        :param InputType letter: Input.
        :return OutputType: Output of the probabilistically chosen transition.
        """
        transition = choice(self.current_state.transitions[letter])
        output = transition[0]
        self.current_state = transition[1]
        return output

    def outputs_on_input(self, letter: InputType) -> list[OutputType]:
        """All possible observable outputs after executing the current input 'letter'.

        :param InputType letter: Input.
        :return list[OutputType]: List of observable outputs.
        """
        return [trans[0] for trans in self.current_state.transitions[letter]]

    def step_to(self, inp: InputType, out: OutputType) -> OutputType | None:
        """Performs a step on the automaton based on the input `inp` and output `out`.

        :param InputType inp: Input.
        :param OutputType out: Output.
        :return OutputType | None: Output of the reached state, None otherwise.
        """
        for new_state in self.current_state.transitions[inp]:
            if new_state[0] == out:
                self.current_state = new_state[1]
                return out
        return None

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'Onfsm':
        """
        Not yet implemented.

        :param dict state_setup: Map from state_id to state configuration.
        """
        raise NotImplementedError()  # TODO implement

    def to_state_setup(self) -> dict:
        """
        Not yet implemented.

        :return dict: Map from state_id to state configuration.
        """
        raise NotImplementedError()  # TODO implement
