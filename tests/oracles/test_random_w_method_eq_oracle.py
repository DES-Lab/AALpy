import random
import unittest

from aalpy.oracles import RandomWMethodEqOracle
from tests.oracles.test_baseOracle import BaseOracleTests


class RandomWMethodEqOracleTests(BaseOracleTests):

    def test_default(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls()

        eq_oracle = RandomWMethodEqOracle(alphabet, learning_sul)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_small_alphabet_more_walks(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(6, 3, 2)

        eq_oracle = RandomWMethodEqOracle(alphabet, learning_sul, walks_per_state=40, walk_len=8)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_finds_cex_over_several_seeds(self):
        successes = 0
        for seed in range(10):
            random.seed(seed)
            learning_sul, validation_sul, alphabet = self.generate_dfa_suls(6, 3, 3)
            reference_dfa = learning_sul.automaton

            broken_model = reference_dfa.copy()
            broken_model.states[-1].is_accepting = not broken_model.states[-1].is_accepting

            oracle = RandomWMethodEqOracle(alphabet, validation_sul, walks_per_state=25, walk_len=12)
            cex = oracle.find_cex(broken_model)
            if cex is not None:
                successes += 1

        self.assertGreaterEqual(successes, 9)

    def test_no_cex_for_equivalent_hypothesis(self):
        random.seed(0)
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(6, 3, 3)
        reference_dfa = learning_sul.automaton

        equivalent_model = reference_dfa.copy()

        oracle = RandomWMethodEqOracle(alphabet, validation_sul, walks_per_state=25, walk_len=12)
        self.assertIsNone(oracle.find_cex(equivalent_model))

    def test_walks_per_state_zero_never_tests_anything(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(6, 3, 3)
        reference_dfa = learning_sul.automaton

        broken_model = reference_dfa.copy()
        broken_model.states[-1].is_accepting = not broken_model.states[-1].is_accepting

        oracle = RandomWMethodEqOracle(alphabet, validation_sul, walks_per_state=0, walk_len=12)
        self.assertIsNone(oracle.find_cex(broken_model))
        self.assertEqual(oracle.num_queries, 0)


if __name__ == '__main__':
    unittest.main()
