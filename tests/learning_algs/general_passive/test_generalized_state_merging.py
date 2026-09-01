import random
import unittest
from itertools import product

from aalpy.automata import Dfa, DfaState, MooreMachine, MooreState, MealyMachine, MealyState
from aalpy.learning_algs.general_passive.GeneralizedStateMerging import (
    GeneralizedStateMerging, Instrumentation, run_GSM,
)
from aalpy.learning_algs.general_passive.GsmNode import GsmNode, unknown_output
from aalpy.utils.HelperFunctions import dfa_from_moore
from aalpy.utils.ModelChecking import bisimilar


def alternating_moore(depth=4):
    """2-state Moore machine over {'a', 'b'}: 'a' toggles state, 'b' stays put."""
    q0 = MooreState('q0', 0)
    q1 = MooreState('q1', 1)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return MooreMachine(q0, [q0, q1])


def parity_mealy():
    q0 = MealyState('q0')
    q1 = MealyState('q1')
    q0.transitions = {'a': q1, 'b': q0}
    q0.output_fun = {'a': 'x', 'b': 'y'}
    q1.transitions = {'a': q0, 'b': q1}
    q1.output_fun = {'a': 'y', 'b': 'x'}
    return MealyMachine(q0, [q0, q1])


def labeled_sequence_data(automaton, depth=3):
    data = []
    alphabet = automaton.get_input_alphabet()
    for level in range(0, depth + 1):
        for seq in product(alphabet, repeat=level):
            automaton.reset_to_initial()
            if len(seq) == 0:
                label = automaton.initial_state.output
            else:
                outputs = automaton.execute_sequence(automaton.initial_state, seq)
                label = outputs[-1]
            data.append((seq, label))
    return data


class RecordingInstrumentation(Instrumentation):
    """Instrumentation that records each hook call, used to check that run_GSM actually invokes them."""

    def __init__(self):
        super().__init__()
        self.reset_called = False
        self.pta_done = False
        self.promotions = []
        self.merges = []
        self.learning_done_called = False

    def reset(self, gsm):
        self.reset_called = True

    def pta_construction_done(self, root):
        self.pta_done = True

    def log_promote(self, node):
        self.promotions.append(node)

    def log_merge(self, part):
        self.merges.append(part)

    def learning_done(self, root):
        self.learning_done_called = True


class TestRunGsmDeterministic(unittest.TestCase):
    def test_learns_correct_moore_machine(self):
        ground_truth = alternating_moore()
        data = labeled_sequence_data(ground_truth, depth=3)
        learned = run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
                          data_format='labeled_sequences')
        self.assertEqual(len(learned.states), 2)
        self.assertTrue(bisimilar(learned, ground_truth))

    def test_learns_correct_mealy_machine(self):
        ground_truth = parity_mealy()
        alphabet = ground_truth.get_input_alphabet()
        traces = []
        for level in range(1, 4):
            for seq in product(alphabet, repeat=level):
                ground_truth.reset_to_initial()
                outputs = ground_truth.execute_sequence(ground_truth.initial_state, seq)
                traces.append(list(zip(seq, outputs)))
        learned = run_GSM(traces, output_behavior='mealy', transition_behavior='deterministic',
                          data_format='io_traces')
        self.assertEqual(len(learned.states), 2)
        self.assertTrue(bisimilar(learned, ground_truth))

    def test_dfa_via_moore_and_dfa_from_moore_conversion(self):
        q0 = DfaState('q0', is_accepting=True)
        q1 = DfaState('q1', is_accepting=False)
        q0.transitions = {'a': q1, 'b': q0}
        q1.transitions = {'a': q0, 'b': q1}
        ground_truth = Dfa(q0, [q0, q1])

        data = labeled_sequence_data(ground_truth, depth=3)
        learned_moore = run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
                                data_format='labeled_sequences')
        learned_dfa = dfa_from_moore(learned_moore)
        self.assertEqual(len(learned_dfa.states), 2)
        self.assertTrue(bisimilar(learned_dfa, ground_truth))

    def test_convert_false_returns_gsm_node(self):
        data = [((), True)]
        result = run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
                         data_format='labeled_sequences', convert=False)
        self.assertIsInstance(result, GsmNode)

    def test_raises_for_invalid_output_behavior(self):
        with self.assertRaises(ValueError):
            GeneralizedStateMerging(output_behavior='invalid')

    def test_raises_for_invalid_transition_behavior(self):
        with self.assertRaises(ValueError):
            GeneralizedStateMerging(transition_behavior='invalid')

    def test_raises_for_nondeterministic_data_with_deterministic_behavior(self):
        gsm = GeneralizedStateMerging(output_behavior='mealy', transition_behavior='deterministic')
        # two different outputs for the same input from the root is nondeterministic
        traces = [[('a', 'x')], [('a', 'y')]]
        with self.assertRaises(ValueError):
            gsm.run(traces, data_format='io_traces')

    def test_missing_score_calc_for_nondeterministic_behavior_raises(self):
        with self.assertRaises(ValueError):
            GeneralizedStateMerging(transition_behavior='nondeterministic')


class TestRunGsmInstrumentation(unittest.TestCase):
    def test_instrumentation_hooks_are_invoked(self):
        ground_truth = alternating_moore()
        data = labeled_sequence_data(ground_truth, depth=3)
        instrumentation = RecordingInstrumentation()

        run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
               data_format='labeled_sequences', instrumentation=instrumentation)

        self.assertTrue(instrumentation.reset_called)
        self.assertTrue(instrumentation.pta_done)
        self.assertTrue(instrumentation.learning_done_called)
        # some merges should have happened since the ground truth is only 2 states
        self.assertGreater(len(instrumentation.merges), 0)


class TestRunGsmPreprocessingPostprocessing(unittest.TestCase):
    def test_preprocessing_and_postprocessing_are_applied(self):
        calls = []

        def pta_preprocessing(root):
            calls.append('pre')
            return root

        def postprocessing(root):
            calls.append('post')
            return root

        data = [((), True), (('a',), False)]
        run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
               data_format='labeled_sequences', pta_preprocessing=pta_preprocessing,
               postprocessing=postprocessing)

        self.assertEqual(calls, ['pre', 'post'])


class TestConsiderOnlyMinBlueAndDepthFirst(unittest.TestCase):
    def test_consider_only_min_blue_still_learns_correct_model(self):
        ground_truth = alternating_moore()
        data = labeled_sequence_data(ground_truth, depth=3)
        learned = run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
                          data_format='labeled_sequences', consider_only_min_blue=True)
        self.assertTrue(bisimilar(learned, ground_truth))

    def test_depth_first_still_learns_correct_model(self):
        ground_truth = alternating_moore()
        data = labeled_sequence_data(ground_truth, depth=3)
        learned = run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
                          data_format='labeled_sequences', depth_first=True,
                          compatibility_on_futures=True)
        self.assertTrue(bisimilar(learned, ground_truth))


if __name__ == '__main__':
    unittest.main()
