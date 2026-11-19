import random
import unittest
from math import ceil, log

from aalpy.automata import MealyMachine, MealyState
from aalpy.oracles import PacOracle
from aalpy.SULs import AutomatonSUL


def chain_mealy(length, alphabet=('a', 'b')):
    """Chain of states over `alphabet`, s0 -> s1 -> ... -> s_length, driven by repeating the first letter."""
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


def with_diverging_transition(mm, diverge_at, alphabet=('a', 'b')):
    mm.states[diverge_at - 1].output_fun[alphabet[0]] = 'x'
    return mm


class PacOracleTests(unittest.TestCase):

    def test_finds_cex_over_several_seeds(self):
        successes = 0
        for seed in range(10):
            random.seed(seed)
            reference = chain_mealy(3)
            hypothesis = chain_mealy(3)
            with_diverging_transition(hypothesis, 3)

            oracle = PacOracle(['a', 'b'], AutomatonSUL(reference), epsilon=0.05, delta=0.05,
                               min_walk_len=5, max_walk_len=10)
            cex = oracle.find_cex(hypothesis)
            if cex is not None:
                successes += 1
                reference.reset_to_initial()
                hypothesis.reset_to_initial()
                sul_out = [reference.step(i) for i in cex]
                hyp_out = [hypothesis.step(i) for i in cex]
                self.assertNotEqual(sul_out[-1], hyp_out[-1])

        self.assertGreaterEqual(successes, 9)

    def test_no_cex_for_equivalent_hypothesis(self):
        # No amount of randomness can turn a truly equivalent hypothesis into a false positive: since every
        # input sequence produces identical output on both models, out_sul == out_hyp always holds, regardless
        # of which random sequences are sampled. This assertion is therefore exact, not merely "usually true".
        random.seed(0)
        reference = chain_mealy(3)
        hypothesis = chain_mealy(3)

        oracle = PacOracle(['a', 'b'], AutomatonSUL(reference), epsilon=0.05, delta=0.05)
        self.assertIsNone(oracle.find_cex(hypothesis))

    def test_number_of_test_cases_grows_with_round_and_shrinks_with_epsilon_delta(self):
        reference = chain_mealy(3)
        hypothesis = chain_mealy(3)

        loose_oracle = PacOracle(['a', 'b'], AutomatonSUL(reference), epsilon=0.5, delta=0.5)
        loose_oracle.find_cex(hypothesis)
        expected_loose = ceil(1 / 0.5 * (log(1 / 0.5) + 1 * log(2)))
        self.assertEqual(loose_oracle.num_queries, expected_loose)

        strict_oracle = PacOracle(['a', 'b'], AutomatonSUL(reference), epsilon=0.02, delta=0.02)
        strict_oracle.find_cex(hypothesis)
        expected_strict = ceil(1 / 0.02 * (log(1 / 0.02) + 1 * log(2)))
        self.assertEqual(strict_oracle.num_queries, expected_strict)

        self.assertGreater(strict_oracle.num_queries, loose_oracle.num_queries)

    def test_round_counter_increases_number_of_test_cases_across_calls(self):
        reference = chain_mealy(3)
        hypothesis = chain_mealy(3)

        oracle = PacOracle(['a', 'b'], AutomatonSUL(reference), epsilon=0.1, delta=0.1)
        oracle.find_cex(hypothesis)
        first_round_queries = oracle.num_queries

        oracle.find_cex(hypothesis)
        second_round_queries = oracle.num_queries - first_round_queries

        self.assertGreater(second_round_queries, first_round_queries)


if __name__ == '__main__':
    unittest.main()
