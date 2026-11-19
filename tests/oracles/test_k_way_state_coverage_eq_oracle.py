import random
import unittest

from aalpy.oracles import KWayStateCoverageEqOracle
from tests.oracles.test_baseOracle import BaseOracleTests


class KWayStateCoverageEqOracleTests(BaseOracleTests):

    def test_default(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls()

        eq_oracle = KWayStateCoverageEqOracle(alphabet, learning_sul)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_k_3(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(6, 5, 3)

        eq_oracle = KWayStateCoverageEqOracle(alphabet, learning_sul, k=3)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_method_combinations(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls()

        eq_oracle = KWayStateCoverageEqOracle(alphabet, learning_sul, method='combinations')
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_lower_and_upper_bounds(self):
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(6, 4, 3)

        eq_oracle = KWayStateCoverageEqOracle(alphabet, learning_sul, num_test_lower_bound=20,
                                              num_test_upper_bound=200)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, validation_sul)

    def test_finds_transition_only_difference(self):
        # a difference that is only reachable via a k-way state combination, not via the random walk tail
        random.seed(0)
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(8, 4, 3)

        # no need to (re-)learn a reference model via run_Lstar here - learning_sul already wraps a
        # correct ground-truth DFA (this used to run a full L* + WMethodEqOracle learning pass just to
        # get back an equivalent copy of the model it already had, taking ~30s for no benefit)
        reference_model = learning_sul.automaton

        # flip acceptance of a non-initial state to create a real, deterministic difference
        target_state = next(s for s in reference_model.states if s is not reference_model.initial_state)
        broken_model = reference_model.copy()
        broken_state = next(s for s in broken_model.states if s.state_id == target_state.state_id)
        broken_state.is_accepting = not broken_state.is_accepting

        oracle = KWayStateCoverageEqOracle(alphabet, validation_sul, k=2)
        cex = oracle.find_cex(broken_model)
        self.assertIsNotNone(cex)

    def test_k_larger_than_hypothesis_state_count_still_tests(self):
        # Regression test: previously, when k exceeded the hypothesis' number of states, both
        # itertools.combinations/permutations produced no k-wise tuples AND the num_test_lower_bound
        # fallback (meant to cover this exact situation) only triggered for single-state hypotheses,
        # so test_cases stayed empty and find_cex returned None without testing anything at all.
        learning_sul, validation_sul, alphabet = self.generate_dfa_suls(6, 4, 3)
        reference_dfa = learning_sul.automaton

        broken_model = reference_dfa.copy()
        broken_model.states = broken_model.states[:2]
        broken_model.initial_state = broken_model.states[0]
        for state in broken_model.states:
            state.transitions = {a: broken_model.states[0] for a in alphabet}

        oracle = KWayStateCoverageEqOracle(alphabet, validation_sul, k=3)
        cex = oracle.find_cex(broken_model)
        self.assertIsNotNone(cex)


if __name__ == '__main__':
    unittest.main()
