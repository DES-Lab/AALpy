# Markov chain state and automaton implementation, where transitions are probabilistic and input-independent.
import random
from collections.abc import Hashable
from typing import Generic

from aalpy.base import Automaton, AutomatonState
from aalpy.base.Automaton import OutputType


class McState(AutomatonState, Generic[OutputType]):
    """
    Single state of a Markov chain. Each state has an output value and a list of probabilistic transitions.
    """

    def __init__(self, state_id: Hashable, output: OutputType) -> None:
        """
        Creates a Markov chain state.

        :param Hashable state_id: Unique identifier of the state.
        :param OutputType output: Output value associated with the state.
        """
        super().__init__(state_id)
        self.output: OutputType = output
        # transitions is a list of tuples (Node(output), probability)
        self.transitions: list[tuple[McState[OutputType], float]] = list()


class MarkovChain(Automaton[McState[OutputType]]):
    """Markov Decision Process."""

    def __init__(self, initial_state: McState, states: list) -> None:
        """
        Creates a Markov chain.

        :param McState initial_state: Initial state of the Markov chain.
        :param list states: All states of the Markov chain.
        """
        super().__init__(initial_state, states)

    def reset_to_initial(self) -> None:
        """
        Resets the current state of the Markov chain to the initial state.
        """
        self.current_state = self.initial_state

    def step(self, letter: None = None) -> OutputType:
        """Next step is determined based on transition probabilities of the current state.

        :param None letter: Unused input, kept for interface compatibility.
        :return OutputType: Output of the reached state.
        """

        if not self.current_state.transitions:
            return self.current_state.output

        probability_distributions = [i[1] for i in self.current_state.transitions]
        states = [i[0] for i in self.current_state.transitions]

        new_state = random.choices(states, probability_distributions, k=1)[0]

        self.current_state = new_state
        return self.current_state.output

    def step_to(self, input: OutputType) -> OutputType | None:
        """Performs a step on the automaton based on the output value to transition to.

        :param OutputType input: Output value identifying the target state among the current state's transitions.
        :return OutputType | None: Output of the reached state, None otherwise.
        """
        for s in self.current_state.transitions:
            if s[0].output == input:
                self.current_state = s[0]
                return self.current_state.output
        return None

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'MarkovChain':
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
