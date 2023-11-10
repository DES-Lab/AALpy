import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import WMethodEqOracle
from aalpy.utils import generate_random_dfa


class BaseOracleTests(unittest.TestCase):
    """
    Abstract class for testing oracles.
    """

    def generate_dfa_suls(self, number_of_states=10, alphabet_size=10, num_accepting_states=5):
        """
        Creates a random DFA and creates a learning and validation SUL, both SUL are identical.

        Args:
            number_of_states: number of states (Default value = 10)
            alphabet_size: size of alphabet (Default value = 10)
            num_accepting_states: number of accepting states (Default value = 5)

        Returns: learning_sul, validation_sul, alphabet

        """
        alphabet = [*range(0, alphabet_size)]

        dfa = generate_random_dfa(number_of_states, alphabet, num_accepting_states)

        learning_sul = AutomatonSUL(dfa)
        validation_sul = AutomatonSUL(dfa)

        return learning_sul, validation_sul, alphabet

    def test_validate_eq_oracle(self, alphabet, eq_oracle, learning_sul, validation_sul):
        """
        Validates the correctness of the given eq_oracle via WMethodEqOracle.

        Args:
            alphabet: input alphabet
            eq_oracle: oracle to be validated
            learning_sul:the SUL form the eq_oracle
            validation_sul: identical SUL that was not used to learn the oracle

        Returns:

        """
        learned_model = run_Lstar(
            alphabet, learning_sul, eq_oracle, 'dfa', print_level=2)

        validation_eq_oracle = WMethodEqOracle(
            alphabet, validation_sul, max_number_of_states=len(learned_model.states) + 2)
        self.assertIsNone(validation_eq_oracle.find_cex(
            learned_model), "Counterexample found by WMethodEqOracle")
