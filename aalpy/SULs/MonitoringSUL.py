# SUL wrapper for performing runtime monitoring.
from typing import Any, Callable, Dict

from aalpy.base import PropertyMonitor, SUL


class MonitoringSUL(SUL):
    """
    SUL wrapper that performs runtime monitoring.
    """

    def __init__(self, sul: SUL,
                 property_monitors: Dict[str,PropertyMonitor],
                 property_violation_callback: Callable[[str,tuple],None] | None = None):
        """
        Create a SUL wrapper that performs runtime monitoring. 

        :param SUL sul: Wrapped SUL.
        :param Dict[str,PropertyMonitor] property_monitors: Dictionary that maps string-typed labels
            to the PropertyMonitors that should be used for monitoring.
        :param Callable[[str,tuple],None] | None property_violation_callback: Optional callback to be
            called when a monitor is found to be violated by the SUL. The monitor's label and the
            counterexample input sequence are then passed to the callback as arguments.
        """
        self.sul = sul

        self.property_monitors = property_monitors
        self.property_violation_callback = property_violation_callback

        # Keep track of the properties for which counterexamples have been found
        self.violated_properties = set()

        # Keep track of the latest input sequence so that it can be passed to the property_violation_callback
        self.inputs = []


    def __getattr__(self, name):
        # Forward attribute/method lookups that MonitoringSUL does not define to the wrapped SUL,
        # so wrappers like Wrapper(SUL) stay accessible through the monitoring layer (e.g. from an equivalence oracle)
        if name == 'sul':
            raise AttributeError(name)
        return getattr(self.sul, name)

    # Bind the number of queries to that of the wrapped system under learning.
    @property
    def num_queries(self) -> int:
        return self.sul.num_queries
    @num_queries.setter
    def num_queries(self, value: int) -> None:
        self.sul.num_queries = value

    # Bind the number of steps to that of the wrapped system under learning.
    @property
    def num_steps(self) -> int:
        return self.sul.num_steps
    @num_steps.setter
    def num_steps(self, value: int) -> None:
        self.sul.num_steps = value

    # Bind the number of cached queries to that of the wrapped system under learning.
    @property
    def num_cached_queries(self) -> int:
        return self.sul.num_cached_queries
    @num_cached_queries.setter
    def num_cached_queries(self, value: int) -> None:
        self.sul.num_cached_queries = value


    def forget_all_property_violations(self) -> None:
        """
        Forgets which properties have been found to be violated by the system under learning, so that
        all properties will be checked again in the next call of self.step.
        """
        self.violated_properties = set()

    def pre(self) -> None:
        """
        Resets the system under learning and the monitors that are still in use.
        """
        self.sul.pre()

        for label, mon in self.property_monitors.items():
            mon.pre()

        self.inputs = []

    def post(self) -> None:
        """
        Performs additional cleanup on the monitor and the monitors that are still in use.
        """
        self.sul.post()

        for label, mon in self.property_monitors.items():
            mon.post()

    def step(self, letter: Any) -> Any:
        """
        Executes an action on the system under learning and returns its result. Also checks the action
        and result against the monitors for which no SUL violations have been found yet.

        :param Any letter: Single input that is executed on the SUL and active monitors.
        :return Any: Output received after executing the input on the SUL.
        """
        # Perform a normal SUL step
        output = self.sul.step(letter)

        self.inputs.append(letter)

        for label, mon in self.property_monitors.items():
            # Skip this property if a SUL counterexample was already found
            if label in self.violated_properties:
                continue

            # Stop monitoring for this property and trigger an optional callback if the monitor doesn't accept the most recent step
            accepted = mon.step(letter, output)
            if not accepted:
                self.violated_properties.add(label)

                if self.property_violation_callback is not None:
                    self.property_violation_callback(label, tuple(self.inputs))

        return output