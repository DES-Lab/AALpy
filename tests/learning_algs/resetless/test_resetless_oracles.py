import random
import unittest

from aalpy.automata import MealyMachine, MealyState, MooreMachine, MooreState
from aalpy.learning_algs.resetless.resetless_oracles import (
    RandomhWOracle,
    RandomWphWOracle,
    find_counterexample_in_trace,
    hWOracle,
)
from aalpy.SULs import AutomatonSUL


def chain_mealy(length, alphabet=('a', 'b')):
    """Chain of states s0 -> s1 -> ... -> s_length, advanced by the first letter of alphabet, looping on the rest."""
    states = [MealyState(f's{i}') for i in range(length + 1)]
    first = alphabet[0]
    for i in range(length):
        states[i].transitions = {a: states[i] for a in alphabet}
        states[i].transitions[first] = states[i + 1]
        states[i].output_fun = {a: 'o' for a in alphabet}
    states[length].transitions = {a: states[length] for a in alphabet}
    states[length].output_fun = {a: 'o' for a in alphabet}
    mm = MealyMachine(states[0], states)
    mm.compute_prefixes()
    return mm


def ring_mealy(size, alphabet=('a', 'b')):
    """
    Ring of `size` states: 'a' advances cyclically s0->s1->...->s(size-1)->s0, 'b' self-loops.
    Strongly connected, so every state can reach every other state (unlike chain_mealy).
    """
    states = [MealyState(f's{i}') for i in range(size)]
    first = alphabet[0]
    for i in range(size):
        states[i].transitions = {a: states[i] for a in alphabet}
        states[i].transitions[first] = states[(i + 1) % size]
        states[i].output_fun = {a: 'o' for a in alphabet}
    mm = MealyMachine(states[0], states)
    mm.compute_prefixes()
    return mm


class FakeLearner:
    """Minimal stand-in for hW exposing just what the resetless oracles/backstop need."""

    def __init__(self, sul, input_alphabet, W=None, is_moore=False):
        self.sul = sul
        self.input_alphabet = input_alphabet
        self.W = W if W is not None else []
        self.is_moore = is_moore
        self.global_trace = []

    def step_wrapper(self, letter):
        output = self.sul.step(letter)
        self.global_trace.append((letter, output))
        return output


class HWOracleBaseTests(unittest.TestCase):

    def test_find_cex_is_not_implemented_on_base_class(self):
        oracle = hWOracle()
        self.assertIsNone(oracle.learner)
        self.assertEqual(oracle.num_steps, 0)
        with self.assertRaises(NotImplementedError):
            oracle.find_cex(hypothesis=None)

    def test_execute_and_compare_stops_at_first_mismatch_and_counts_steps(self):
        reference = chain_mealy(3)
        hypothesis = chain_mealy(3)
        hypothesis.states[1].output_fun['a'] = 'x'

        sul = AutomatonSUL(reference)
        sul.pre()
        learner = FakeLearner(sul, ['a', 'b'])
        oracle = hWOracle()
        oracle.learner = learner

        cex = []
        mismatched = oracle._execute_and_compare(hypothesis, ('a', 'a', 'a'), cex)

        self.assertTrue(mismatched)
        # the reference walks s0->s1->s2 while the hypothesis is stepped in lock-step;
        # the divergent output_fun edit is only observed on the second 'a' (from s1)
        self.assertEqual(cex, ['a', 'a'])
        self.assertEqual(oracle.num_steps, 2)

    def test_execute_and_compare_returns_false_when_no_mismatch(self):
        reference = chain_mealy(3)
        hypothesis = chain_mealy(3)

        sul = AutomatonSUL(reference)
        sul.pre()
        learner = FakeLearner(sul, ['a', 'b'])
        oracle = hWOracle()
        oracle.learner = learner

        cex = []
        mismatched = oracle._execute_and_compare(hypothesis, ('a', 'b', 'a'), cex)

        self.assertFalse(mismatched)
        self.assertEqual(cex, ['a', 'b', 'a'])
        self.assertEqual(oracle.num_steps, 3)


class RandomhWOracleTests(unittest.TestCase):

    def test_finds_cex_over_several_seeds(self):
        successes = 0
        for seed_val in range(10):
            random.seed(seed_val)
            reference = chain_mealy(6)
            hypothesis = chain_mealy(6)
            hypothesis.states[6].output_fun['a'] = 'x'

            sul = AutomatonSUL(reference)
            sul.pre()
            learner = FakeLearner(sul, ['a', 'b'])
            oracle = RandomhWOracle(num_testing_steps=500)
            oracle.learner = learner

            cex = oracle.find_cex(hypothesis)
            if cex is not None:
                successes += 1
                self.assertEqual(cex[-1], 'a')

        self.assertGreaterEqual(successes, 9)

    def test_no_cex_for_equivalent_hypothesis(self):
        random.seed(0)
        reference = chain_mealy(4)
        hypothesis = chain_mealy(4)

        sul = AutomatonSUL(reference)
        sul.pre()
        learner = FakeLearner(sul, ['a', 'b'])
        oracle = RandomhWOracle(num_testing_steps=500)
        oracle.learner = learner

        self.assertIsNone(oracle.find_cex(hypothesis))
        self.assertEqual(oracle.num_steps, 500)

    def test_reset_testing_counter_false_shares_budget_across_calls(self):
        random.seed(0)
        reference = chain_mealy(4)
        hypothesis = chain_mealy(4)

        sul = AutomatonSUL(reference)
        sul.pre()
        learner = FakeLearner(sul, ['a', 'b'])
        oracle = RandomhWOracle(num_testing_steps=10, reset_testing_counter=False)
        oracle.learner = learner

        oracle.find_cex(hypothesis)
        self.assertEqual(oracle.num_steps, 10)

        # the whole budget of 10 was already consumed by the first call, so a
        # second call must execute zero further steps
        oracle.find_cex(hypothesis)
        self.assertEqual(oracle.num_steps, 10)

    def test_reset_testing_counter_true_replenishes_budget_each_call(self):
        random.seed(0)
        reference = chain_mealy(4)
        hypothesis = chain_mealy(4)

        sul = AutomatonSUL(reference)
        sul.pre()
        learner = FakeLearner(sul, ['a', 'b'])
        oracle = RandomhWOracle(num_testing_steps=10, reset_testing_counter=True)
        oracle.learner = learner

        oracle.find_cex(hypothesis)
        oracle.find_cex(hypothesis)
        self.assertEqual(oracle.num_steps, 20)


class RandomWphWOracleTests(unittest.TestCase):

    def test_finds_cex_over_several_seeds(self):
        successes = 0
        for seed_val in range(10):
            random.seed(seed_val)
            reference = ring_mealy(4)
            hypothesis = ring_mealy(4)
            hypothesis.states[2].output_fun['b'] = 'x'

            sul = AutomatonSUL(reference)
            sul.pre()
            learner = FakeLearner(sul, ['a', 'b'], W=[('a',), ('b',)])
            oracle = RandomWphWOracle(random_walk_length=10, num_test_origin_states=8)
            oracle.learner = learner

            cex = oracle.find_cex(hypothesis)
            if cex is not None:
                successes += 1

        self.assertGreaterEqual(successes, 9)

    def test_no_cex_for_equivalent_hypothesis(self):
        random.seed(0)
        reference = chain_mealy(4)
        hypothesis = chain_mealy(4)

        sul = AutomatonSUL(reference)
        sul.pre()
        learner = FakeLearner(sul, ['a', 'b'], W=[('a',), ('b',)])
        oracle = RandomWphWOracle(random_walk_length=5, num_test_origin_states=5)
        oracle.learner = learner

        self.assertIsNone(oracle.find_cex(hypothesis))

    def test_empty_characterization_set_does_not_crash(self):
        random.seed(0)
        reference = chain_mealy(3)
        hypothesis = chain_mealy(3)

        sul = AutomatonSUL(reference)
        sul.pre()
        learner = FakeLearner(sul, ['a', 'b'], W=[])
        oracle = RandomWphWOracle(random_walk_length=5, num_test_origin_states=3)
        oracle.learner = learner

        self.assertIsNone(oracle.find_cex(hypothesis))

    def test_zero_origin_states_never_tests_anything(self):
        reference = chain_mealy(3)
        hypothesis = chain_mealy(3)
        hypothesis.states[2].output_fun['a'] = 'x'

        sul = AutomatonSUL(reference)
        sul.pre()
        learner = FakeLearner(sul, ['a', 'b'], W=[('a',)])
        oracle = RandomWphWOracle(random_walk_length=5, num_test_origin_states=0)
        oracle.learner = learner

        self.assertIsNone(oracle.find_cex(hypothesis))
        self.assertEqual(oracle.num_steps, 0)


class FindCounterexampleInTraceTests(unittest.TestCase):

    def test_returns_none_for_trace_fully_explained_by_hypothesis(self):
        hypothesis = chain_mealy(3)
        learner = FakeLearner(sul=None, input_alphabet=['a', 'b'], is_moore=False)
        learner.global_trace = [('a', 'o'), ('a', 'o'), ('b', 'o')]

        self.assertIsNone(find_counterexample_in_trace(learner, hypothesis))

    def test_finds_unexplained_suffix_for_mealy(self):
        hypothesis = chain_mealy(3)
        learner = FakeLearner(sul=None, input_alphabet=['a', 'b'], is_moore=False)
        # 'x' is never produced by any state of the hypothesis
        learner.global_trace = [('a', 'o'), ('a', 'x')]

        cex = find_counterexample_in_trace(learner, hypothesis)

        self.assertIsNotNone(cex)
        self.assertEqual(cex[-1], 'a')

    def test_finds_unexplained_step_for_moore(self):
        s0 = MooreState('s0', output='0')
        s1 = MooreState('s1', output='1')
        s0.transitions = {'a': s1}
        s1.transitions = {'a': s0}
        mm = MooreMachine(s0, [s0, s1])

        learner = FakeLearner(sul=None, input_alphabet=['a'], is_moore=True)
        # after one 'a' the hypothesis would be in s1 (output '1'), not '9'
        learner.global_trace = [('a', '9')]

        cex = find_counterexample_in_trace(learner, mm)
        self.assertEqual(cex, ['a'])


if __name__ == '__main__':
    unittest.main()
