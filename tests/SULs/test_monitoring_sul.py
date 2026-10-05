from typing import Any, Tuple
import unittest

from aalpy.automata import Dfa, DfaState, MealyState, MealyMachine
from aalpy.base import PropertyMonitor
from aalpy.property_monitors import IUOBugDfaMonitor
from aalpy.SULs import AutomatonSUL, MonitoringSUL


def get_accepting_monitor() -> PropertyMonitor:
    """
    Create an Oracle which never returns a counterexample.
    """
    class AcceptingMonitor(PropertyMonitor):
        def __init__(self):
            super().__init__()

            self.pre_called = False
            self.post_called = False
            self.step_taken = False

        def pre(self):
            self.pre_called = True

        def post(self):
            self.post_called = True

        def step(self, input: Any, output: Any) -> bool:
            self.step_taken = True
            return True

    return AcceptingMonitor()


def precise_word_mealy(word: Tuple[str], alphabet=('a', 'b')):
    """
    Mealy machine that outputs len(word) - 1 `o`s followed by an `x` and then only 'o's for the word
    and only `o`s for any other input sequence.
    """
    length = len(word)
    sink = MealyState('sink')
    sink.transitions = {a: sink for a in alphabet}
    sink.output_fun = {a: 'o' for a in alphabet}
    states = [MealyState(f's_{word[:i]}') for i in range(length)] + [sink]
    for i in range(length):
        states[i].transitions = {a: sink for a in alphabet}
        states[i].output_fun = {a: 'o' for a in alphabet}
        if i < length - 1:
            states[i].transitions[word[i]] = states[i + 1]
        else:
            states[i].output_fun[word[i]] = 'x'
    return MealyMachine(states[0], states)

def precise_word_dfa(word: Tuple[str], alphabet=('a', 'b')):
    """
    Dfa which accepts the input sequences that are prefixed by the given word.
    """
    length = len(word)
    sink = DfaState('sink', False)
    sink.transitions = {a: sink for a in alphabet}
    states = [DfaState(f's_{word[:i]}', i == length) for i in range(length + 1)] + [sink]
    for i in range(length):
        states[i].transitions = {a: sink for a in alphabet}
        states[i].transitions[word[i]] = states[i + 1]
    states[length].transitions = {a: sink for a in alphabet}
    return Dfa(states[0], states)

def get_precise_word_adaptive_query(accepted_input_word: tuple, expected_output_word: tuple) -> Any:
    """
    Create an adaptive query which provides precisely the accepted_input_word and verifies that it
    receives the expected_output_word.
    """
    class PreciseWordADS:
        def __init__(self, accepted_input_word: tuple, expected_output_word: tuple):
            self.remaining_accepted_input_word = accepted_input_word
            self.remaining_expected_output_word = expected_output_word

        def next_input(self, last_output: Any) -> Any:
            if last_output is not None:
                assert len(self.remaining_expected_output_word) > 0
                assert self.remaining_expected_output_word[0] == last_output
                self.remaining_expected_output_word = self.remaining_expected_output_word[1:]

            if len(self.remaining_accepted_input_word) == 0:
                return None

            next_in = self.remaining_accepted_input_word[0]
            self.remaining_accepted_input_word = self.remaining_accepted_input_word[1:]

            return next_in

    return PreciseWordADS(accepted_input_word, expected_output_word)


def dfa_input_from_mealy_input(input: Any) -> Tuple[str,Any]:
    return ('I', input)

def dfa_output_from_mealy_output(output: Any) -> Tuple[str,Any]:
    return ('O', output)


class MonitoringSULTests(unittest.TestCase):

    def test_values_are_bound_properly(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        base_sul = AutomatonSUL(mealy)

        sul = MonitoringSUL(
            sul=base_sul,
            property_monitors=dict()
        )

        self.assertEqual(sul.num_queries, base_sul.num_queries)
        self.assertEqual(sul.num_steps, base_sul.num_steps)
        self.assertEqual(sul.num_cached_queries, base_sul.num_cached_queries)

        sul.num_queries = 12345
        sul.num_steps = 54321
        sul.num_cached_queries = 123454321
        self.assertEqual(base_sul.num_queries, sul.num_queries)
        self.assertEqual(base_sul.num_steps, sul.num_steps)
        self.assertEqual(base_sul.num_cached_queries, sul.num_cached_queries)

    def test_base_sul_steps_are_taken_when_no_properties(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        base_sul = AutomatonSUL(mealy)

        sul = MonitoringSUL(
            sul=base_sul,
            property_monitors=dict()
        )

        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'o')
        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'x')

    def test_no_prop_cex_found_if_none_exist(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        base_sul = AutomatonSUL(mealy)

        def violation_callback(label: str, cex: tuple) -> None:
            # The property has no counterexample that could have been confirmed against the SUT, so the callback may not have been called with a counterexample
            assert False

        sul = MonitoringSUL(
            sul=base_sul,
            property_monitors={
                'SatisfiedProperty': get_accepting_monitor()
            },
            property_violation_callback=violation_callback
        )

        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'o')
        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'x')

    def test_prop_cex_found_if_and_only_if_it_exist(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        base_sul = AutomatonSUL(mealy)

        violation_cex = None
        def violation_callback(label: str, cex: tuple) -> None:
            nonlocal violation_cex

            self.assertEqual(label, 'ViolatedProperty')
            violation_cex = cex

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
        violated_prop_monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output
        )

        sul = MonitoringSUL(
            sul=base_sul,
            property_monitors={
                'SatisfiedProperty': get_accepting_monitor(),
                'ViolatedProperty': violated_prop_monitor
            },
            property_violation_callback=violation_callback
        )

        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'o')
        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'x')
        self.assertIsNotNone(violation_cex)

        # The violated_prop_monitor should yield a False for precisely the end of violation_cex 
        sul.pre()
        violated_prop_monitor.pre()
        for i, input in enumerate(violation_cex):
            output = sul.step(input)
            verdict = violated_prop_monitor.step(input, output)

            if i < len(violation_cex) - 1:
                self.assertTrue(verdict)
            else:
                self.assertFalse(verdict)

    def test_violation_callback_is_optional(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        base_sul = AutomatonSUL(mealy)

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
        violated_prop_monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output
        )

        sul = MonitoringSUL(
            sul=base_sul,
            property_monitors={
                'SatisfiedProperty': get_accepting_monitor(),
                'ViolatedProperty': violated_prop_monitor
            }
        )

        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'o')
        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'x')

    def test_properties_with_counterexamples_are_excluded_until_reenabled(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        base_sul = AutomatonSUL(mealy)

        violation_cex = None
        def violation_callback(label: str, cex: tuple) -> None:
            nonlocal violation_cex

            self.assertEqual(label, 'ViolatedProperty')
            violation_cex = cex

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
        violated_prop_monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output
        )

        sul = MonitoringSUL(
            sul=base_sul,
            property_monitors={
                'SatisfiedProperty': get_accepting_monitor(),
                'ViolatedProperty': violated_prop_monitor
            },
            property_violation_callback=violation_callback
        )

        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'o')
        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'x')
        self.assertIsNotNone(violation_cex)
        violation_cex = None

        sul.pre()
        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'o')
        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'x')
        self.assertIsNone(violation_cex)

        sul.pre()
        sul.forget_all_property_violations()        
        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'o')
        self.assertEqual(sul.step('a'), 'o')
        self.assertEqual(sul.step('b'), 'x')
        self.assertIsNotNone(violation_cex)

    def test_pre_calls_sul_and_prop_pres(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        base_sul = AutomatonSUL(mealy)

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
        violated_prop_monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output
        )

        satisfied_prop_monitor = get_accepting_monitor()
        sul = MonitoringSUL(
            sul=base_sul,
            property_monitors={
                'SatisfiedProperty': satisfied_prop_monitor,
                'ViolatedProperty': violated_prop_monitor
            }
        )

        sul.step('a')
        self.assertNotEqual(sul.sul.automaton.current_state, sul.sul.automaton.initial_state)
        self.assertFalse(satisfied_prop_monitor.pre_called)
        self.assertNotEqual(violated_prop_monitor.bug_dfa.current_state, violated_prop_monitor.bug_dfa.initial_state)

        sul.pre()
        self.assertEqual(sul.sul.automaton.current_state, sul.sul.automaton.initial_state)
        self.assertTrue(satisfied_prop_monitor.pre_called)
        self.assertEqual(violated_prop_monitor.bug_dfa.current_state, violated_prop_monitor.bug_dfa.initial_state)

    def test_post_calls_sul_and_prop_posts(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        base_sul = AutomatonSUL(mealy)

        satisfied_prop_monitor = get_accepting_monitor()
        sul = MonitoringSUL(
            sul=base_sul,
            property_monitors={
                'SatisfiedProperty': satisfied_prop_monitor
            }
        )

        sul.step('a')
        self.assertNotEqual(sul.sul.automaton.current_state, sul.sul.automaton.initial_state)
        self.assertFalse(satisfied_prop_monitor.post_called)

        sul.post()
        self.assertNotEqual(sul.sul.automaton.current_state, sul.sul.automaton.initial_state)
        self.assertTrue(satisfied_prop_monitor.post_called)

    def test_queries_and_steps_are_counted_correctly_for_standard_queries(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        base_sul = AutomatonSUL(mealy)

        violation_cex_num_queries = None
        violation_cex_num_steps = None
        violation_cex_num_cached_queries = None
        def violation_callback(label: str, cex: tuple) -> None:
            nonlocal violation_cex_num_queries
            nonlocal violation_cex_num_steps
            nonlocal violation_cex_num_cached_queries

            self.assertEqual(label, 'ViolatedProperty')
            violation_cex = cex

            violation_cex_num_queries = base_sul.num_queries
            violation_cex_num_steps = base_sul.num_steps
            violation_cex_num_cached_queries = base_sul.num_cached_queries

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
        violated_prop_monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output
        )

        sul = MonitoringSUL(
            sul=base_sul,
            property_monitors={
                'SatisfiedProperty': get_accepting_monitor(),
                'ViolatedProperty': violated_prop_monitor
            },
            property_violation_callback=violation_callback
        )

        # Perform a first query which finds the property violation at the very end
        first_word = ('a', 'b', 'a', 'b')
        expected_first_word_outputs = ('o', 'o', 'o', 'x')
        first_word_outputs = sul.query(first_word)
        self.assertIsNotNone(first_word_outputs)
        self.assertEqual(tuple(first_word_outputs), expected_first_word_outputs)

        self.assertEqual(sul.num_queries, 1)
        self.assertEqual(sul.num_steps, len(first_word))
        self.assertEqual(sul.num_cached_queries, 0)

        self.assertEqual(violation_cex_num_queries, 1)
        self.assertEqual(violation_cex_num_steps, len(first_word))
        self.assertEqual(violation_cex_num_cached_queries, 0)

        sul.forget_all_property_violations()

        # Perform a second query which finds the property violation before the end is reached
        second_word = ('a', 'b', 'a', 'b', 'a', 'a')
        expected_second_word_outputs = ('o', 'o', 'o', 'x', 'o', 'o')
        second_word_outputs = sul.query(second_word)
        self.assertIsNotNone(second_word_outputs)
        self.assertEqual(tuple(second_word_outputs), expected_second_word_outputs)

        self.assertEqual(sul.num_queries, 2)
        self.assertEqual(sul.num_steps, len(first_word) + len(second_word))
        self.assertEqual(sul.num_cached_queries, 0)

        self.assertEqual(violation_cex_num_queries, 2)
        self.assertEqual(violation_cex_num_steps, len(first_word) * 2)
        self.assertEqual(violation_cex_num_cached_queries, 0)

    def test_queries_and_steps_are_counted_correctly_for_adaptive_queries(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        base_sul = AutomatonSUL(mealy)

        violation_cex_num_queries = None
        violation_cex_num_steps = None
        violation_cex_num_cached_queries = None
        def violation_callback(label: str, cex: tuple) -> None:
            nonlocal violation_cex_num_queries
            nonlocal violation_cex_num_steps
            nonlocal violation_cex_num_cached_queries

            self.assertEqual(label, 'ViolatedProperty')
            violation_cex = cex

            violation_cex_num_queries = base_sul.num_queries
            violation_cex_num_steps = base_sul.num_steps
            violation_cex_num_cached_queries = base_sul.num_cached_queries

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
        violated_prop_monitor = IUOBugDfaMonitor(
            bug_dfa=prop_bug_dfa,
            to_dfa_input=dfa_input_from_mealy_input,
            to_dfa_output=dfa_output_from_mealy_output
        )

        sul = MonitoringSUL(
            sul=base_sul,
            property_monitors={
                'SatisfiedProperty': get_accepting_monitor(),
                'ViolatedProperty': violated_prop_monitor
            },
            property_violation_callback=violation_callback
        )

        # Perform a first adaptive query which finds the property violation partway through the standard word
        first_word = ('a', 'b', 'a', 'b', 'a', 'a', 'b')
        expected_first_word_outputs = ('o', 'o', 'o', 'x', 'o', 'o', 'o')
        first_word_ads = get_precise_word_adaptive_query(first_word[5:], expected_first_word_outputs[5:])
        (first_word_performed_inputs, first_word_outputs) = sul.adaptive_query(list(first_word[:5]), first_word_ads)
        self.assertEqual(tuple(first_word_performed_inputs), first_word)
        self.assertIsNotNone(first_word_outputs)
        self.assertEqual(tuple(first_word_outputs), expected_first_word_outputs)

        self.assertEqual(sul.num_queries, 1)
        self.assertEqual(sul.num_steps, len(first_word))
        self.assertEqual(sul.num_cached_queries, 0)

        self.assertEqual(violation_cex_num_queries, 1)
        self.assertEqual(violation_cex_num_steps, 4)
        self.assertEqual(violation_cex_num_cached_queries, 0)

        sul.forget_all_property_violations()

        # Perform a second adaptive query which finds the property violation partway through the adaptive word
        second_word = ('a', 'b', 'a', 'b', 'a', 'a')
        expected_second_word_outputs = ('o', 'o', 'o', 'x', 'o', 'o')
        second_word_ads = get_precise_word_adaptive_query(second_word[2:], expected_second_word_outputs[2:])
        (second_word_performed_inputs, second_word_outputs) = sul.adaptive_query(list(second_word[:2]), second_word_ads)
        self.assertEqual(tuple(second_word_performed_inputs), second_word)
        self.assertIsNotNone(second_word_outputs)
        self.assertEqual(tuple(second_word_outputs), expected_second_word_outputs)

        self.assertEqual(sul.num_queries, 2)
        self.assertEqual(sul.num_steps, len(first_word) + len(second_word))
        self.assertEqual(sul.num_cached_queries, 0)

        self.assertEqual(violation_cex_num_queries, 2)
        self.assertEqual(violation_cex_num_steps, len(first_word) + 4)
        self.assertEqual(violation_cex_num_cached_queries, 0)


if __name__ == '__main__':
    unittest.main()
