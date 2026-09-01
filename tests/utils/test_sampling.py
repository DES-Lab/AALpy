import random
import unittest

from aalpy.automata import Dfa, DfaState, MealyMachine, MealyState
from aalpy.utils.Sampling import (
    get_io_traces,
    get_labeled_sequences,
    get_data_from_input_sequence,
    sample_with_length_limits,
    sample_with_term_prob,
    get_complete_sample,
)


def parity_dfa():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    dfa = Dfa(q0, [q0, q1])
    dfa.compute_prefixes()
    return dfa


def parity_mealy():
    q0 = MealyState('q0')
    q1 = MealyState('q1')
    q0.transitions = {'a': q1, 'b': q0}
    q0.output_fun = {'a': 'x', 'b': 'y'}
    q1.transitions = {'a': q0, 'b': q1}
    q1.output_fun = {'a': 'y', 'b': 'x'}
    mealy = MealyMachine(q0, [q0, q1])
    mealy.compute_prefixes()
    return mealy


class TestGetIoTraces(unittest.TestCase):
    def test_dfa_traces_prefixed_with_initial_output(self):
        dfa = parity_dfa()
        traces = get_io_traces(dfa, [['a', 'a'], ['b']])
        self.assertEqual(traces[0][0], True)
        self.assertEqual(traces[0][1:], [('a', False), ('a', True)])
        self.assertEqual(traces[1], [True, ('b', True)])

    def test_mealy_traces_not_prefixed(self):
        mealy = parity_mealy()
        traces = get_io_traces(mealy, [['a', 'b']])
        self.assertEqual(traces[0], [('a', 'x'), ('b', 'x')])

    def test_empty_input_trace(self):
        # regression test: Dfa.execute_sequence (like MooreMachine's) returns a bare output value
        # rather than a list for an empty sequence, which used to make zip(input_trace, output_trace)
        # crash with "'bool' object is not iterable".
        dfa = parity_dfa()
        traces = get_io_traces(dfa, [[]])
        self.assertEqual(traces[0], [True])


class TestGetLabeledSequences(unittest.TestCase):
    def test_dfa_labels(self):
        dfa = parity_dfa()
        data = get_labeled_sequences(dfa, [['a'], ['a', 'a']])
        self.assertEqual(data, [(['a'], False), (['a', 'a'], True)])

    def test_dfa_empty_sequence_returns_initial_output(self):
        dfa = parity_dfa()
        data = get_labeled_sequences(dfa, [[]])
        self.assertEqual(data, [([], True)])

    def test_mealy_empty_sequence_raises(self):
        mealy = parity_mealy()
        with self.assertRaises(ValueError):
            get_labeled_sequences(mealy, [[]])


class TestGetDataFromInputSequence(unittest.TestCase):
    def test_io_sequences_format(self):
        dfa = parity_dfa()
        data = get_data_from_input_sequence(dfa, [['a']], data_format='io_sequences')
        self.assertEqual(data, get_io_traces(dfa, [['a']]))

    def test_labeled_sequences_format(self):
        dfa = parity_dfa()
        data = get_data_from_input_sequence(dfa, [['a']], data_format='labeled_sequences')
        self.assertEqual(data, get_labeled_sequences(dfa, [['a']]))

    def test_invalid_format_raises(self):
        dfa = parity_dfa()
        with self.assertRaises(ValueError):
            get_data_from_input_sequence(dfa, [['a']], data_format='bogus')


class TestSampleWithLengthLimits(unittest.TestCase):
    def test_alphabet_argument(self):
        random.seed(1)
        samples = sample_with_length_limits(['a', 'b'], nr_samples=10, min_len=2, max_len=5)
        self.assertEqual(len(samples), 10)
        for s in samples:
            self.assertTrue(2 <= len(s) <= 5)
            self.assertTrue(all(x in {'a', 'b'} for x in s))

    def test_automaton_argument_uses_input_alphabet(self):
        random.seed(2)
        dfa = parity_dfa()
        samples = sample_with_length_limits(dfa, nr_samples=5, min_len=1, max_len=3)
        for s in samples:
            self.assertTrue(all(x in {'a', 'b'} for x in s))

    def test_include_outputs_requires_automaton(self):
        with self.assertRaises(ValueError):
            sample_with_length_limits(['a', 'b'], nr_samples=1, min_len=1, max_len=1, include_outputs=True)

    def test_include_outputs_with_automaton(self):
        random.seed(3)
        dfa = parity_dfa()
        samples = sample_with_length_limits(dfa, nr_samples=3, min_len=1, max_len=2, include_outputs=True)
        for trace in samples:
            self.assertEqual(trace[0], True)
            for i, o in trace[1:]:
                self.assertIn(i, {'a', 'b'})
                self.assertIn(o, {True, False})


class TestSampleWithTermProb(unittest.TestCase):
    def test_generates_requested_number_of_samples(self):
        random.seed(4)
        samples = sample_with_term_prob(['a', 'b'], nr_samples=8, term_prob=0.5)
        self.assertEqual(len(samples), 8)
        for s in samples:
            self.assertTrue(all(x in {'a', 'b'} for x in s))

    def test_term_prob_one_gives_empty_sequences(self):
        random.seed(5)
        samples = sample_with_term_prob(['a', 'b'], nr_samples=5, term_prob=1.0)
        for s in samples:
            self.assertEqual(s, [])


class TestGetCompleteSample(unittest.TestCase):
    def test_complete_sample_covers_alphabet_states_and_char_set(self):
        dfa = parity_dfa()
        sample = get_complete_sample(dfa)
        self.assertTrue(len(sample) > 0)
        char_set = dfa.compute_characterization_set()
        for seq in sample:
            self.assertIsInstance(seq, tuple)
        self.assertEqual(len(sample), len(dfa.states) * len(char_set) * (len(dfa.get_input_alphabet()) + 1))

    def test_accepts_automaton_arg_via_decorator(self):
        random.seed(6)
        dfa = parity_dfa()
        sample = get_complete_sample(dfa)
        self.assertTrue(all(isinstance(s, tuple) for s in sample))


if __name__ == '__main__':
    unittest.main()
