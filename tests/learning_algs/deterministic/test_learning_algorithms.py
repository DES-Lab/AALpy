import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Dfa, MealyMachine, MooreMachine
from aalpy.learning_algs import run_KV, run_Lsharp, run_Lstar
from aalpy.oracles import RandomWalkEqOracle, RandomWMethodEqOracle, WMethodEqOracle
from aalpy.utils import get_Angluin_dfa, load_automaton_from_file
from aalpy.utils.ModelChecking import bisimilar
from pathlib import Path

DOT_MODELS_DIR = Path(__file__).resolve().parents[3] / 'DotModels'


def reference_automaton(automaton_type):
    if automaton_type == 'dfa':
        return get_Angluin_dfa()
    if automaton_type == 'mealy':
        return load_automaton_from_file(str(DOT_MODELS_DIR / 'Angluin_Mealy.dot'), automaton_type='mealy')
    if automaton_type == 'moore':
        return load_automaton_from_file(str(DOT_MODELS_DIR / 'Angluin_Moore.dot'), automaton_type='moore')
    raise ValueError(automaton_type)


def learns_correctly(learning_alg, automaton_type, **kwargs):
    """
    Builds a fresh SUL/oracle around the reference Angluin automaton of the requested type, runs the
    given learning algorithm with the given config, and asserts the learned hypothesis is minimal and
    bisimilar to the ground truth (which itself is already known to be minimal for all three types).
    """
    ground_truth = reference_automaton(automaton_type)
    alphabet = ground_truth.get_input_alphabet()
    sul = AutomatonSUL(ground_truth)
    eq_oracle = RandomWMethodEqOracle(alphabet, sul, walks_per_state=50, walk_len=20)

    learned_model = learning_alg(alphabet, sul, eq_oracle, automaton_type=automaton_type, print_level=0, **kwargs)

    assert learned_model.is_minimal(), f'{learning_alg.__name__}/{automaton_type}: learned model is not minimal'
    assert bisimilar(ground_truth, learned_model), \
        f'{learning_alg.__name__}/{automaton_type}: learned model is not bisimilar to ground truth'
    return learned_model


class TestRunLstarConfigurations(unittest.TestCase):
    AUTOMATON_TYPES = ['dfa', 'mealy', 'moore']

    def test_default_configuration(self):
        for automaton_type in self.AUTOMATON_TYPES:
            learns_correctly(run_Lstar, automaton_type)

    def test_closing_strategies(self):
        for closing in ['shortest_first', 'longest_first', 'single']:
            learns_correctly(run_Lstar, 'dfa', closing_strategy=closing)

    def test_cex_processing_strategies(self):
        for cex_processing in [None, 'rs', 'longest_prefix', 'linear_fwd', 'linear_bwd',
                                'exponential_fwd', 'exponential_bwd']:
            learns_correctly(run_Lstar, 'mealy', cex_processing=cex_processing)

    def test_suffix_closedness_options(self):
        for suffix_closed in [True, False]:
            learns_correctly(run_Lstar, 'moore', all_prefixes_in_obs_table=True, e_set_suffix_closed=suffix_closed)

    def test_without_caching_and_non_det_check(self):
        learns_correctly(run_Lstar, 'dfa', cache_and_non_det_check=False)

    def test_with_a_different_equivalence_oracle(self):
        ground_truth = reference_automaton('dfa')
        alphabet = ground_truth.get_input_alphabet()
        sul = AutomatonSUL(ground_truth)
        eq_oracle = WMethodEqOracle(alphabet, sul, max_number_of_states=len(ground_truth.states) + 1)
        learned_model = run_Lstar(alphabet, sul, eq_oracle, automaton_type='dfa', print_level=0)
        self.assertTrue(learned_model.is_minimal())
        self.assertTrue(bisimilar(ground_truth, learned_model))

    def test_return_data_reports_consistent_learning_rounds(self):
        ground_truth = reference_automaton('dfa')
        alphabet = ground_truth.get_input_alphabet()
        sul = AutomatonSUL(ground_truth)
        eq_oracle = RandomWalkEqOracle(alphabet, sul, 1000)
        learned_model, info = run_Lstar(alphabet, sul, eq_oracle, automaton_type='dfa', print_level=0,
                                        return_data=True)
        self.assertTrue(bisimilar(ground_truth, learned_model))
        self.assertGreaterEqual(info['learning_rounds'], 1)
        self.assertEqual(info['automaton_size'], len(learned_model.states))


class TestRunKvConfigurations(unittest.TestCase):
    AUTOMATON_TYPES = ['dfa', 'mealy', 'moore']

    def test_default_configuration(self):
        for automaton_type in self.AUTOMATON_TYPES:
            learns_correctly(run_KV, automaton_type)

    def test_cex_processing_strategies(self):
        for cex_processing in ['rs', 'linear_fwd', 'linear_bwd', 'exponential_fwd', 'exponential_bwd']:
            learns_correctly(run_KV, 'mealy', cex_processing=cex_processing)

    def test_without_caching_and_non_det_check(self):
        learns_correctly(run_KV, 'dfa', cache_and_non_det_check=False)

    def test_return_data_reports_consistent_learning_rounds(self):
        ground_truth = reference_automaton('moore')
        alphabet = ground_truth.get_input_alphabet()
        sul = AutomatonSUL(ground_truth)
        eq_oracle = RandomWMethodEqOracle(alphabet, sul, walks_per_state=50, walk_len=20)
        learned_model, info = run_KV(alphabet, sul, eq_oracle, automaton_type='moore', print_level=0,
                                     return_data=True)
        self.assertTrue(bisimilar(ground_truth, learned_model))
        self.assertEqual(info['automaton_size'], len(learned_model.states))


class TestRunLsharpConfigurations(unittest.TestCase):
    AUTOMATON_TYPES = ['dfa', 'mealy', 'moore']

    def test_default_configuration(self):
        for automaton_type in self.AUTOMATON_TYPES:
            learns_correctly(run_Lsharp, automaton_type)

    def test_extension_and_separation_rule_combinations(self):
        for extension_rule in [None, 'SepSeq', 'ADS']:
            for separation_rule in ['SepSeq', 'ADS']:
                learns_correctly(run_Lsharp, 'dfa', extension_rule=extension_rule, separation_rule=separation_rule)

    def test_extension_and_separation_rules_on_mealy_and_moore(self):
        for automaton_type in ['mealy', 'moore']:
            learns_correctly(run_Lsharp, automaton_type, extension_rule='ADS', separation_rule='ADS')

    def test_without_caching_and_non_det_check(self):
        learns_correctly(run_Lsharp, 'dfa', cache_and_non_det_check=False)

    def test_return_data_reports_consistent_learning_rounds(self):
        ground_truth = reference_automaton('dfa')
        alphabet = ground_truth.get_input_alphabet()
        sul = AutomatonSUL(ground_truth)
        eq_oracle = RandomWMethodEqOracle(alphabet, sul, walks_per_state=50, walk_len=20)
        learned_model, info = run_Lsharp(alphabet, sul, eq_oracle, automaton_type='dfa', print_level=0,
                                         return_data=True)
        self.assertTrue(bisimilar(ground_truth, learned_model))
        self.assertEqual(info['automaton_size'], len(learned_model.states))


if __name__ == '__main__':
    unittest.main()
