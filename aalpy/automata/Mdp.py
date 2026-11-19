# Markov decision process state and automaton implementation, where transitions are probabilistic given an input.
import random
from collections import defaultdict
from collections.abc import Hashable
from typing import Generic

from aalpy.base import Automaton, AutomatonState
from aalpy.base.Automaton import OutputType, InputType


class MdpState(AutomatonState, Generic[InputType, OutputType]):
    """
    Single state of an MDP. For transitions, each transition is a tuple (Node(output), probability).
    """

    def __init__(self, state_id: Hashable, output: OutputType | None = None) -> None:
        """
        Creates an MDP state.

        :param Hashable state_id: Unique identifier of the state.
        :param OutputType | None output: Output value associated with the state.
        """
        super().__init__(state_id)
        self.output: OutputType | None = output
        # each transition is a tuple (Node(output), probability)
        self.transitions: dict[InputType, list[tuple[MdpState[InputType, OutputType], float]]] = defaultdict(list)


class Mdp(Automaton[MdpState[InputType, OutputType]]):
    """Markov Decision Process."""

    def __init__(self, initial_state: MdpState, states: list) -> None:
        """
        Creates an MDP.

        :param MdpState initial_state: Initial state of the MDP.
        :param list states: All states of the MDP.
        """
        super().__init__(initial_state, states)

    def reset_to_initial(self) -> None:
        """
        Resets the current state of the MDP to the initial state.
        """
        self.current_state = self.initial_state

    def step(self, letter: InputType | None) -> OutputType:
        """Next step is determined based on transition probabilities of the current state.

        :param InputType | None letter: Input.
        :return OutputType: Output of the current state.
        """
        if letter is None:
            return self.current_state.output

        probability_distributions = [i[1] for i in self.current_state.transitions[letter]]
        states = [i[0] for i in self.current_state.transitions[letter]]

        new_state = random.choices(states, probability_distributions, k=1)[0]

        self.current_state = new_state
        return self.current_state.output

    def step_to(self, inp: InputType, out: OutputType) -> OutputType | None:
        """Performs a step on the automaton based on the input `inp` and output `out`.

        :param InputType inp: Input.
        :param OutputType out: Output.
        :return OutputType | None: Output of the reached state, None otherwise.
        """
        for new_state in self.current_state.transitions[inp]:
            if new_state[0].output == out:
                self.current_state = new_state[0]
                return out
        return None

    def to_state_setup(self) -> dict:
        """
        Converts the MDP to a state setup dictionary.

        :return dict: Map from state_id to tuple(output, transitions_dict).
        """
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
    def from_state_setup(state_setup: dict, **kwargs) -> 'Mdp':
        """
        Creates an MDP from a state setup dictionary. The first state in the state setup is the initial state.

        :param dict state_setup: Map from state_id to tuple(output, transitions_dict).
        :return Mdp: The constructed MDP.
        """
        states_map = {key: MdpState(key, output=value[0]) for key, value in state_setup.items()}

        for key, values in state_setup.items():
            source = states_map[key]
            for i, transitions in values[1].items():
                for node, prob in transitions:
                    source.transitions[i].append((states_map[node], prob))

        initial_state = states_map[list(state_setup.keys())[0]]
        return Mdp(initial_state, list(states_map.values()))
