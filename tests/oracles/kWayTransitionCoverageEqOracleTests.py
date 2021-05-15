import unittest

from aalpy.oracles import KWayTransitionCoverageEqOracle
from baseOracleTests import BaseOracleTests


class KWayTransitionCoverageEqOracleTests(BaseOracleTests):

    def test_default(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls()

        eq_oracle = KWayTransitionCoverageEqOracle(alphabet, learning_sul)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_k_4(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(5, 5, 2)

        eq_oracle = KWayTransitionCoverageEqOracle(
            alphabet, learning_sul, k=4)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_minimize_paths_true(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls()

        eq_oracle = KWayTransitionCoverageEqOracle(
            alphabet, learning_sul, minimize_paths=True)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_refills_0(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls()

        eq_oracle = KWayTransitionCoverageEqOracle(
            alphabet, learning_sul, refills=0)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    @unittest.expectedFailure
    def test_target_coverage_0(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls()

        eq_oracle = KWayTransitionCoverageEqOracle(
            alphabet, learning_sul, target_coverage=0)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_default_large_dfa(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(50, 10, 10)

        eq_oracle = KWayTransitionCoverageEqOracle(alphabet, learning_sul)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)
