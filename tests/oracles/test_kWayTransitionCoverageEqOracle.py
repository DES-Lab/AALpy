import unittest

from aalpy.oracles import KWayTransitionCoverageEqOracle
from tests.oracles.test_baseOracle import BaseOracleTests


class KWayTransitionCoverageEqOracleTests(BaseOracleTests):

    def test_default(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls()

        eq_oracle = KWayTransitionCoverageEqOracle(alphabet, learning_sul)
        self.test_validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_k_4(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(5, 5, 2)

        eq_oracle = KWayTransitionCoverageEqOracle(
            alphabet, learning_sul, k=4)
        self.test_validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_method_prefix(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls()

        eq_oracle = KWayTransitionCoverageEqOracle(
            alphabet, learning_sul, method='prefix')
        self.test_validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    @unittest.expectedFailure
    def test_max_number_of_steps_10(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(50, 4, 4)

        eq_oracle = KWayTransitionCoverageEqOracle(
            alphabet, learning_sul, max_number_of_steps=10, max_path_len=10)
        self.test_validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_default_large_dfa(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(50, 10, 10)

        eq_oracle = KWayTransitionCoverageEqOracle(alphabet, learning_sul)
        self.test_validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)
