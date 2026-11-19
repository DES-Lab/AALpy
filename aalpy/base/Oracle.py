# Abstract base class for all equivalence oracles.
from abc import ABC, abstractmethod

from aalpy.base import SUL
from aalpy.base.Automaton import Automaton, InputType


class Oracle(ABC):
    """Abstract class implemented by all equivalence oracles."""

    def __init__(self, alphabet: list, sul: SUL) -> None:
        """
        Default constructor for all equivalence oracles.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        """

        self.alphabet = alphabet
        self.sul = sul
        self.num_queries = 0
        self.num_steps = 0

    @abstractmethod
    def find_cex(self, hypothesis: Automaton) -> tuple[InputType, ...] | None:
        """
        Return a counterexample (inputs) that displays different behavior on system under learning and
        current hypothesis.

        :param Automaton hypothesis: Current hypothesis.
        :return tuple[InputType, ...] | None: Counterexample inputs, None if no counterexample is found.
        """
        pass

    def reset_hyp_and_sul(self, hypothesis: Automaton) -> None:
        """
        Reset SUL and hypothesis to initial state.

        :param Automaton hypothesis: Current hypothesis.
        """
        hypothesis.reset_to_initial()
        self.sul.pre()
        self.num_queries += 1
