# Property monitor that checks for given Mealy machines with input alphabet I and output alphabet O whether their language is rejected by a Dfa with alphabet I union O.
from aalpy.automata import Dfa
from aalpy.base import PropertyMonitor
from typing import Any, Callable


class IUOBugDfaMonitor(PropertyMonitor):
    """
    Monitor that determines for given steps with inputs from alpabet I and outputs from alphabet O
    whether the traces they describe are allowed by a given property. This property is represented
    by a Dfa which encodes traces over the union of I and O which violate the property.
    """

    def __init__(self, bug_dfa: Dfa,
                 to_dfa_input: Callable[[Any],Any],
                 to_dfa_output: Callable[[Any],Any],
                 bug_dfa_fallback_letter: Any = None,
                 report_non_final_violations: bool = True
                ):
        """
        Create a monitor which checks whether the traces formed by the steps taken in the monitor
        are accepted by a Dfa which accepts traces over the union of the Mealy machine's inputs and
        outputs which violate some property.

        :param Dfa bug_dfa: Dfa that accepts the traces which violate some property.
        :param Callable[[Any],Any] to_dfa_input: Function that converts inputs provided in monitor
            steps into letters accepted by the bug_dfa.
        :param Callable[[Any],Any] to_dfa_output: Function that converts outputs provided in monitor
            steps into letters accepted by the bug_dfa.
        :param Any bug_dfa_fallback_letter: Optional letter in the bug_dfa's alphabet for which the
            bug_dfa should perform a state transition whenever an input or output provided for a
            monitor step cannot be matched to the letters for any of its other state transitions.
        :param bool report_non_final_violations: Report a property violation if the first of the two
            bug_dfa states encountered for a step is accepting, regardless of whether the second/final
            encountered bug_dfa state is accepting.
        """
        super().__init__()
        self.bug_dfa = bug_dfa
        self.to_dfa_input = to_dfa_input
        self.to_dfa_output = to_dfa_output
        self.bug_dfa_fallback_letter = bug_dfa_fallback_letter
        self.report_non_final_violations = report_non_final_violations

    def pre(self) -> None:
        """
        Reset self.bug_dfa to its initial state.
        """
        self.bug_dfa.reset_to_initial()

    def post(self) -> None:
        """
        Perform no cleanup as this isn't required by self.bug_dfa.
        """
        pass

    def step(self, input: Any, output: Any) -> bool:
        """
        Perform state transitions on self.bug_dfa for a given input and output and return True
        if and only if self.bug_dfa rejects these transitions.

        :param Any input: Input.
        :param Any output: Output.
        :return bool: Whether the property represented by self.bug_dfa accepts the step.
        """
        violation = self._take_bug_dfa_step(self.to_dfa_input(input))
        if violation and self.report_non_final_violations:
            # Still perform the second step in self.bug_dfa, so that it ends up in the correct state
            self._take_bug_dfa_step(self.to_dfa_output(output))

            return False

        violation = self._take_bug_dfa_step(self.to_dfa_output(output))

        return not violation

    def _take_bug_dfa_step(self, letter: Any) -> bool:
        """
        Take a single step in self.bug_dfa.

        :param Any letter: Letter for which self.bug_dfa should perform a state transition from its
            current state.
        :return bool: Whether the reached state is accepted by self.bug_dfa, and therefore rejected
            by the property represented by self.bug_dfa.
        """
        if letter in self.bug_dfa.current_state.transitions:
            return self.bug_dfa.step(letter)
        else:
            if self.bug_dfa_fallback_letter is None:
                raise ValueError(f"Bug DFA state `{self.bug_dfa.current_state.state_id}' doesn't have a transition for the letter `{letter}' and no fallback letter was provided")
            elif self.bug_dfa_fallback_letter not in self.bug_dfa.current_state.transitions:
                raise ValueError(f"Bug DFA state `{self.bug_dfa.current_state.state_id}' doesn't have a transition for either the letter `{letter}' or the fallback letter `{self.bug_dfa_fallback_letter}'")

            return self.bug_dfa.step(self.bug_dfa_fallback_letter)