from __future__ import annotations

import string
from collections import defaultdict
from copy import deepcopy
from random import choice

from aalpy.base import Automaton, AutomatonState


class IotsState(AutomatonState):
    def __init__(self, state_id):
        super().__init__(state_id)
        # Note: inputs/outputs maps to tuples of possible new state e.g. input => (state1, state2)
        self.inputs = defaultdict(tuple)
        self.outputs = defaultdict(tuple)
        # Note: workaround for ioco
        self.ioco_status = None

    def get_inputs(self, input: string = None, destination: IotsState = None) -> list[tuple[str, IotsState]]:
        assert input is None or input.startswith('?')

        result = [(input, state)
                  for input, states in self.inputs.items() for state in states]
        result = result if input is None else list(
            filter(lambda elm: elm[0] == input, result))
        result = result if destination is None else list(
            filter(lambda elm: elm[1] == destination, result))

        return result

    def get_outputs(self, output: string = None, destination: IotsState = None) -> list[tuple[str, IotsState]]:
        assert output is None or output.startswith('!')

        result = [(output, state)
                  for output, states in self.outputs.items() for state in states]
        result = result if output is None else list(
            filter(lambda elm: elm[0] == output, result))
        result = result if destination is None else list(
            filter(lambda elm: elm[1] == destination, result))

        return result

    def add_input(self, input: string, new_state: IotsState):
        assert input.startswith('?')

        new_value = tuple(
            [new_state]) + self.inputs[input] if input in self.inputs else tuple([new_state])
        self.inputs.update({input: new_value})
        self.transitions.update(self.inputs)

    def add_output(self, output: string, new_state):
        assert output.startswith('!')

        new_value = tuple(
            new_state) + self.outputs[output] if output in self.outputs else tuple([new_state])
        self.outputs.update({output: new_value})
        self.transitions.update(self.outputs)

    def is_input_enabled(self) -> bool:
        """
        A state is input enabled if an input can trigger an transition.

        Returns:
            bool: input enabled flag
        """
        return any(self.inputs.values())

    def is_input_enabled_for_diff_state(self) -> bool:
        return all(self not in states for states in self.inputs.values())

    def is_quiescence(self) -> bool:
        """
        A state is quiescence if no output transition exists.

        Returns:
            bool: quiescence flag
        """
        return not any(self.outputs.values())

    def is_deterministic(self) -> bool:
        deterministic_input = all(
            len(states) == 1 for states in self.inputs.values())
        deterministic_output = all(
            len(states) == 1 for states in self.outputs.values())
        return deterministic_input and deterministic_output


class IotsMachine(Automaton):
    """
    Input output transition system machine.
    """

    def __init__(self, initial_state: IotsState, states: list[IotsState]):
        super().__init__(initial_state, states)
        self.current_state: IotsState
        self.initial_state: IotsState
        self.states: list[IotsState]

    def step(self, input: string) -> None:
        """
        Next step is determined based on a uniform distribution over all transitions with the input 'letter'.

        TODO
        I am not sure if the step() function should also work for outputs given by the caller,
        so the user can chose if the automaton steps on an input or output edge. Maybe we trigger
        always an output on the destination state, this would works very well if we restrict
        the automaton to an strict input-output chain (Martin recommend that for the beginning).
        """

        assert input.startswith('?')
        result: list[str] = []
        visited = []

        def input_step(input: str):
            transitions = self.current_state.get_inputs(input)
            if not transitions:
                return None

            (key, self.current_state) = choice(transitions)
            return key

        def output_step(output: str = None):
            transitions = self.current_state.get_outputs(output)
            if not transitions:
                return None

            (key, self.current_state) = choice(transitions)
            return key

        if input_step(input) is None:
            return None

        while True:

            if self.current_state.is_input_enabled():
                break

            if self.current_state.is_quiescence():
                break

            if self.current_state in visited:
                break

            visited.append(self.current_state)
            result.append(output_step())

        return result

    def get_input_alphabet(self) -> list:
        """
        Returns the input alphabet
        """
        result: list[str] = []
        for state in self.states:
            result.extend([input for input, _ in state.get_inputs()])

        return list(set(result))

    def get_output_alphabet(self) -> list:
        """
        Returns the output alphabet
        """
        result: list[str] = []
        for state in self.states:
            result.extend([output for output, _ in state.get_outputs()])

        return list(set(result))


class IocoValidator:

    def __init__(self, specification: IotsMachine):
        self.specification: IotsMachine = deepcopy(specification)

        self.states = []
        self.visited = []
        self.state_count = 0

        self.initial_state = self._new_test_state()
        self._resolve_state(self.specification.initial_state, self.initial_state)

        self.automata: IotsMachine = IotsMachine(self.initial_state, self.states)

    def _new_test_state(self):
        self.state_count += 1
        state = IotsState(f't{self.state_count}')
        state.ioco_status = "test"
        self.states.append(state)
        return state

    def _new_passed_state(self):
        state = self._new_test_state()
        state.ioco_status = "passed"

        state.state_id += " passed"
        return state

    def _new_failed_state(self):
        state = self._new_test_state()
        state.ioco_status = "failed"

        state.state_id += " failed"
        return state

    def _resolve_state(self, original_state: IotsState, test_state: IotsState):
        self.visited.append(original_state)

        follow_state = dict()

        for input in self.specification.get_input_alphabet():
            states = original_state.inputs[input]

            # TODO can a input go to a passed stated? or does it need to go to quiescence state first.
            # TODO can an output follow and output without and input between or vis versa.
            # TODO can the test case have two quiescence transition to the passed state?

            if not states:
                test_state.add_output(input.replace("?", "!"), self._new_failed_state())
            else:
                new_test_state = self._new_test_state()
                result = self._resolve_outputs(new_test_state, states)

                follow_state.update(result)
                test_state.add_output(input.replace("?", "!"), new_test_state)

        self._resolve_outputs(test_state, [original_state])

        for specification_state, ioco_state in follow_state.items():
            if specification_state not in self.visited:
                self._resolve_state(specification_state, ioco_state)

    def _resolve_outputs(self, test_state, states) -> dict:
        follow_state = dict()
        for destination in states:

            if destination.is_quiescence() and destination.is_input_enabled_for_diff_state() and destination not in self.visited:
                new_test_state = self._new_test_state()
                test_state.add_input("?quiescence", new_test_state)
                follow_state.update({destination: new_test_state})
            elif destination.is_quiescence() and destination.is_input_enabled_for_diff_state():
                test_state.add_input("?quiescence", self._new_passed_state())
            else:
                test_state.add_input("?quiescence", self._new_failed_state())

            for output in self.specification.get_output_alphabet():
                transitions = destination.outputs[output]

                if not transitions:
                    test_state.add_input(output.replace("!", "?"), self._new_failed_state())

                for state in transitions:
                    if state.is_quiescence() and not state.is_input_enabled_for_diff_state():
                        test_state.add_input(output.replace("!", "?"), self._new_passed_state())
                    else:
                        new_test_state = self._new_test_state()
                        test_state.add_input(output.replace("!", "?"), new_test_state)
                        follow_state.update({state: new_test_state})

        return follow_state

    def check(self, sut: IotsMachine) -> bool:

        return False
