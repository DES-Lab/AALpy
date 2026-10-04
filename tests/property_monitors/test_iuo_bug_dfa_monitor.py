from typing import Any, Tuple
import unittest

from aalpy.automata import Dfa, DfaState
from aalpy.property_monitors import IUOBugDfaMonitor
from aalpy.utils.ModelChecking import bisimilar


def precise_word_dfa(word: Tuple[str], alphabet=('a', 'b')):
    """
    Dfa which accepts the input sequences that are prefixed by the given word.
    """
    length = len(word)
    sink = DfaState("sink", False)
    sink.transitions = {a: sink for a in alphabet}
    states = [DfaState(f's_{word[:i]}', i == length) for i in range(length + 1)] + [sink]
    for i in range(length):
        states[i].transitions = {a: sink for a in alphabet}
        states[i].transitions[word[i]] = states[i + 1]
    states[length].transitions = {a: sink for a in alphabet}
    return Dfa(states[0], states)


def dfa_input_from_mealy_input(input: Any) -> Tuple[str,Any]:
    return ("I", input)

def dfa_output_from_mealy_output(output: Any) -> Tuple[str,Any]:
    return ("O", output)

def is_dfa_input(letter: Tuple[str,Any]) -> bool:
    return letter[0] == "I"

def mealy_letter_from_dfa_letter(letter: Tuple[str,Any]) -> Any:
    return letter[1]


class IUOBugDfaMonitorTests(unittest.TestCase):

    def test_disallowed_step_is_rejected(self):
        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_output_from_mealy_output('x')
            )
        )

        monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output
        )

        self.assertTrue(monitor.step('a', 'o'))
        self.assertTrue(monitor.step('b', 'o'))
        self.assertTrue(monitor.step('a', 'o'))
        self.assertFalse(monitor.step('b', 'x'))

    def test_rejection_can_be_established_with_fallback_letter(self):
        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x'),
                dfa_output_from_mealy_output('fallback_o')
            )
        )

        monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output,
            bug_dfa_fallback_letter=dfa_output_from_mealy_output('fallback_o')
        )

        self.assertTrue(monitor.step('a', 'o'))
        self.assertTrue(monitor.step('b', 'o'))
        self.assertTrue(monitor.step('a', 'o'))
        self.assertFalse(monitor.step('b', 'x'))

    def test_counterexample_cannot_be_found_when_fallback_output_is_needed_and_missing(self):
        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x'),
                dfa_output_from_mealy_output('fallback_o')
            )
        )

        monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output,
        )

        with self.assertRaises(ValueError):
            monitor.step('a', 'o')

    def test_pre_resets_bug_dfa(self):
        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_output_from_mealy_output('x')
            )
        )

        monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output
        )

        self.assertTrue(monitor.step('a', 'o'))
        monitor.pre()
        self.assertTrue(monitor.step('a', 'o'))
        self.assertTrue(monitor.step('b', 'o'))
        self.assertTrue(monitor.step('a', 'o'))
        self.assertFalse(monitor.step('b', 'x'))

    def test_report_on_final_violations(self):
        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o')
            )
        )

        monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output,
            report_non_final_violations=True
        )

        self.assertTrue(monitor.step('a', 'o'))
        # The 'b'-step leads to an accepting state, upon which the monitor should
        # return False
        self.assertFalse(monitor.step('b', 'o'))
        # Either way, the transitions for both 'b' and 'o' should be performed, so
        # monitor.bug_dfa should end up in a non-accepting sink state.
        self.assertFalse(monitor.bug_dfa.current_state.is_accepting)

    def test_dont_report_on_final_violations(self):
        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o')
            )
        )

        monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output,
            report_non_final_violations=False
        )

        self.assertTrue(monitor.step('a', 'o'))
        # The 'b'-step leads to an accepting state, but the monitor should ignore
        # this and perform the 'o'-step, which leads to a non-accepting sink state
        self.assertTrue(monitor.step('b', 'o'))
        # Either way, the transitions for both 'b' and 'o' should be performed, so
        # monitor.bug_dfa should end up in a non-accepting sink state.
        self.assertFalse(monitor.bug_dfa.current_state.is_accepting)


if __name__ == '__main__':
    unittest.main()
