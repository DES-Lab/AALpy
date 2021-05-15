import unittest

from aalpy.automata import MealyMachine, MooreMachine
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import (CacheBasedEqOracle, KWayStateCoverageEqOracle,
                           KWayTransitionCoverageEqOracle, RandomWalkEqOracle,
                           RandomWMethodEqOracle, RandomWordEqOracle,
                           StatePrefixEqOracle, TransitionFocusOracle,
                           WMethodEqOracle)
from aalpy.SULs import MealySUL, MooreSUL, DfaSUL
from aalpy.utils import generate_random_mealy_machine, generate_random_dfa


class OracleTest(unittest.TestCase):

    def get_random_mealy(self):
        number_of_states = 50
        alphabet_size = 40
        output_size = 90

        alphabet = [*range(0, alphabet_size)]

        result = generate_random_mealy_machine(number_of_states, alphabet, output_alphabet=list(range(output_size)))

        return result, alphabet

    def get_random_dfa(self):
        number_of_states = 50
        alphabet_size = 40
        output_size = 90

        alphabet = [*range(0, alphabet_size)]

        result = generate_random_dfa(number_of_states, alphabet, 10)
        return result, alphabet

    def validate_eq_oracle(self, alphabet, eq_oracle, learning_sul, testing_sul):
        learned_model = run_Lstar(
            alphabet, learning_sul, eq_oracle, 'dfa', print_level=2)

        testing_eq_oracle = RandomWalkEqOracle(alphabet, testing_sul)
        self.assertIsNone(testing_eq_oracle.find_cex(
            learned_model), "Counterexample found by RandomWalkEqOracle")

        testing_eq_oracle = WMethodEqOracle(
            alphabet, testing_sul, max_number_of_states=len(learned_model.states) + 10)
        self.assertIsNone(testing_eq_oracle.find_cex(
            learned_model), "Counterexample found by WMethodEqOracle")

    def test_KWayTransitionCoverageEqOracle(self):
        random_dfa, alphabet = self.get_random_dfa()

        learning_sul = DfaSUL(random_dfa)
        testing_sul = DfaSUL(random_dfa)

        eq_oracle = KWayTransitionCoverageEqOracle(alphabet, learning_sul)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, testing_sul)

        eq_oracle = KWayTransitionCoverageEqOracle(
            alphabet, learning_sul, minimize_paths=True)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, testing_sul)

        eq_oracle = KWayTransitionCoverageEqOracle(
            alphabet, learning_sul, refills=0)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, testing_sul)

        eq_oracle = KWayTransitionCoverageEqOracle(
            alphabet, learning_sul, target_coverage=0.001)
        self.validate_eq_oracle(alphabet, eq_oracle, learning_sul, testing_sul)

        print("KWayTransitionCoverageEqOracle passed")
