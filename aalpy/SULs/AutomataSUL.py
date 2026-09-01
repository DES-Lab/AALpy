# SUL wrapping an in-memory automaton, used to expose learned/reference automata as systems under learning.
from typing import Any

from aalpy.base import Automaton
from aalpy.base import SUL


class AutomatonSUL(SUL):
    """
    System under learning that wraps an in-memory automaton, delegating steps to it.
    """

    def __init__(self, automaton: Automaton) -> None:
        """
        Creates a SUL wrapping an automaton.

        :param Automaton automaton: The automaton to wrap.
        """
        super().__init__()
        self.automaton: Automaton = automaton

    def pre(self) -> None:
        """
        Resets the wrapped automaton to its initial state.
        """
        self.automaton.reset_to_initial()

    def step(self, letter: Any = None) -> Any:
        """
        Executes a single input on the wrapped automaton.

        :param Any letter: Single input that is executed on the wrapped automaton.
        :return Any: Output received after executing the input.
        """
        return self.automaton.step(letter)

    def post(self) -> None:
        """
        Performs no cleanup, as the wrapped automaton requires none between queries.
        """
        pass


MealySUL = OnfsmSUL = StochasticMealySUL = DfaSUL = MooreSUL = MdpSUL = McSUL = SevpaSUL = AutomatonSUL
