# SUL wrapper that records every observed input/output trace into a TraceTree, used by ONFSM learning algorithms.
from typing import Any

from aalpy.base import SUL
from aalpy.learning_algs.non_deterministic.TraceTree import TraceTree


class NonDeterministicSULWrapper(SUL):
    """
    Wrapper for non-deterministic SUL. After every step, input/output pair is added to the tree containing all traces.
    """

    def __init__(self, sul: SUL) -> None:
        """
        Creates a wrapper around a non-deterministic SUL that records all observed traces.

        :param SUL sul: The wrapped system under learning.
        """
        super().__init__()
        self.sul = sul
        self.cache = TraceTree()

    def pre(self) -> None:
        """
        Resets the trace tree cursor and the wrapped system under learning.
        """
        self.cache.reset()
        self.sul.pre()

    def post(self) -> None:
        """
        Performs cleanup on the wrapped system under learning.
        """
        self.sul.post()

    def step(self, letter: Any) -> Any:
        """
        Executes an action on the wrapped system under learning, records it in the trace tree and returns its result.

        :param Any letter: Single input that is executed on the SUL.
        :return Any: Output received after executing the input.
        """
        out = self.sul.step(letter)
        self.cache.add_to_tree(letter, out)
        return out
