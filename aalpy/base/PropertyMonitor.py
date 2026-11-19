# Abstract base class for all property monitors.
from abc import ABC, abstractmethod
from typing import Any


class PropertyMonitor(ABC):
    """
    Interface for objects that can perform runtime monitoring over input-output traces performed by a
    SUL. These objects represent safety properties, in the sense that any counterexample is finite in
    length. The monitor's stepper function returns True for every step that is (still) allowed in the
    property and False for any step that is not allowed.
    """

    @abstractmethod
    def pre(self) -> None:
        """
        Resets the monitor.
        """
        pass

    @abstractmethod
    def post(self) -> None:
        """
        Performs additional cleanup on the monitor if necessary.
        """
        pass

    @abstractmethod
    def step(self, input: Any, output: Any) -> bool:
        """
        Executes an action on the monitor and returns its verdict.

        :param Any input: Single SUL input for the monitor.
        :param Any output: Single SUL output for the monitor.
        :return bool: Whether the monitor accepts the input and output at this point along the
            input-output trace performed since the last invocation of self.pre().
        """
        pass