import unittest

from aalpy.automata import MealyMachine, MealyState
from aalpy.oracles import ProvidedSequencesOracleWrapper
from aalpy.SULs import AutomatonSUL


def sample_mealy():
    """2-state Mealy machine over {x, y}, matching tests/automata/test_mealy_machine.py's fixture."""
    s0 = MealyState('s0')
    s1 = MealyState('s1')
    s0.transitions = {'x': s1, 'y': s0}
    s0.output_fun = {'x': 'o1', 'y': 'o2'}
    s1.transitions = {'x': s0, 'y': s1}
    s1.output_fun = {'x': 'o3', 'y': 'o1'}
    mm = MealyMachine(s0, [s0, s1])
    mm.compute_prefixes()
    return mm


def sample_mealy_with_wrong_output():
    mm = sample_mealy()
    mm.states[0].output_fun['y'] = 'wrong'
    return mm


class RecordingOracle:
    """Never finds a counterexample, records whether it was ever invoked."""

    def __init__(self):
        self.called = False
        self.num_queries = 0
        self.num_steps = 0

    def find_cex(self, hypothesis):
        self.called = True
        return None


class ProvidedSequencesOracleWrapperTests(unittest.TestCase):

    def test_finds_cex_among_provided_sequences(self):
        reference = sample_mealy()
        hypothesis = sample_mealy_with_wrong_output()

        fallback = RecordingOracle()
        oracle = ProvidedSequencesOracleWrapper(['x', 'y'], AutomatonSUL(reference), fallback,
                                                [['x', 'x'], ['y'], ['x', 'y', 'x']])
        cex = oracle.find_cex(hypothesis)

        self.assertEqual(tuple(cex), ('y',))
        self.assertFalse(fallback.called, "fallback oracle should not run once a provided sequence finds a cex")

    def test_no_cex_when_none_of_the_provided_sequences_reveal_it_and_delegates_to_wrapped_oracle(self):
        reference = sample_mealy()
        hypothesis = sample_mealy_with_wrong_output()

        # none of these sequences ever exercise s0's 'y' transition, so they can't reveal the difference
        fallback = RecordingOracle()
        oracle = ProvidedSequencesOracleWrapper(['x', 'y'], AutomatonSUL(reference), fallback,
                                                [['x'], ['x', 'x']])
        cex = oracle.find_cex(hypothesis)

        self.assertIsNone(cex)
        self.assertTrue(fallback.called, "wrapped oracle should be used once provided sequences are exhausted")

    def test_exactly_the_provided_sequences_are_checked_and_only_once(self):
        reference = sample_mealy()
        hypothesis = sample_mealy()

        fallback = RecordingOracle()
        provided = [['x'], ['y'], ['x', 'y']]
        oracle = ProvidedSequencesOracleWrapper(['x', 'y'], AutomatonSUL(reference), fallback, provided)

        self.assertIsNone(oracle.find_cex(hypothesis))
        self.assertEqual(provided, [], "all provided sequences should be consumed")
        self.assertTrue(fallback.called)

    def test_non_revealing_sequences_are_removed_before_the_one_that_finds_the_cex(self):
        # sequences are consumed in order; once one of them reveals the difference, find_cex returns
        # immediately, so that sequence (and anything after it) is left untouched in the provided list.
        reference = sample_mealy()
        hypothesis = sample_mealy_with_wrong_output()

        fallback = RecordingOracle()
        provided = [['x', 'x'], ['y'], ['x', 'y', 'x']]
        oracle = ProvidedSequencesOracleWrapper(['x', 'y'], AutomatonSUL(reference), fallback, provided)
        oracle.find_cex(hypothesis)

        self.assertEqual(provided, [['y'], ['x', 'y', 'x']])


if __name__ == '__main__':
    unittest.main()
