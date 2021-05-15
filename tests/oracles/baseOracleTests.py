import unittest

from aalpy.SULs import DfaSUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import WMethodEqOracle
from aalpy.utils import generate_random_dfa


class BaseOracleTests(unittest.TestCase):

    def generate_dfa_suls(self, number_of_states=10, alphabet_size=10, num_accepting_states=5):
        alphabet = [*range(0, alphabet_size)]

        dfa = generate_random_dfa(number_of_states, alphabet, num_accepting_states)

        learning_sul = DfaSUL(dfa)
        validation_sul = DfaSUL(dfa)

        return learning_sul, validation_sul, alphabet

    def validate_eq_oracle(self, alphabet, eq_oracle, learning_sul, validation_sul):
        learned_model = run_Lstar(
            alphabet, learning_sul, eq_oracle, 'dfa', print_level=2)

        validation_eq_oracle = WMethodEqOracle(
            alphabet, validation_sul, max_number_of_states=len(learned_model.states) + 2)
        self.assertIsNone(validation_eq_oracle.find_cex(
            learned_model), "Counterexample found by WMethodEqOracle")
