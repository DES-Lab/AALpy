import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Dfa, MealyMachine, MooreMachine
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import WMethodEqOracle, WpMethodEqOracle, RandomWalkEqOracle, StatePrefixEqOracle, TransitionFocusOracle, \
    RandomWMethodEqOracle, BreadthFirstExplorationEqOracle, RandomWordEqOracle, CacheBasedEqOracle, \
    KWayStateCoverageEqOracle, RandomWpMethodEqOracle
from aalpy.utils import get_Angluin_dfa, load_automaton_from_file
from aalpy.utils.ModelChecking import bisimilar

correct_automata = {Dfa: get_Angluin_dfa(),
                    MealyMachine: load_automaton_from_file('../DotModels/Angluin_Mealy.dot', automaton_type='mealy'),
                    MooreMachine: load_automaton_from_file('../DotModels/Angluin_Moore.dot', automaton_type='moore')}


class DeterministicTest(unittest.TestCase):

    def prove_equivalence(self, learned_automaton):

        correct_automaton = correct_automata[learned_automaton.__class__]

        # only work if correct automaton is already minimal
        if len(learned_automaton.states) != len(correct_automaton.states):
            print(len(learned_automaton.states), len(correct_automaton.states))
            return False

        return bisimilar(correct_automaton, learned_automaton)

    def test_closing_strategies(self):

        dfa = get_Angluin_dfa()

        alphabet = dfa.get_input_alphabet()

        closing_strategies = ['shortest_first', 'longest_first', 'single']
        automata_type = ['dfa', 'mealy', 'moore']

        for automata in automata_type:
            for closing in closing_strategies:
                sul = AutomatonSUL(dfa)
                eq_oracle = RandomWalkEqOracle(alphabet, sul, 1000)

                learned_dfa = run_Lstar(alphabet, sul, eq_oracle, automaton_type=automata, closing_strategy=closing,
                                        cache_and_non_det_check=True, cex_processing='rs', print_level=0)

                is_eq = self.prove_equivalence(learned_dfa)
                if not is_eq:
                    assert False

        assert True

    def test_suffix_closedness(self):

        angluin_example = get_Angluin_dfa()

        alphabet = angluin_example.get_input_alphabet()

        suffix_closedness = [True, False]
        automata_type = ['dfa', 'mealy', 'moore']

        for automata in automata_type:
            for s_closed in suffix_closedness:
                sul = AutomatonSUL(angluin_example)
                eq_oracle = RandomWalkEqOracle(alphabet, sul, 500)

                learned_dfa = run_Lstar(alphabet, sul, eq_oracle, automaton_type=automata,
                                        all_prefixes_in_obs_table=True,
                                        e_set_suffix_closed=s_closed,
                                        cache_and_non_det_check=True, cex_processing='rs', print_level=0)

                is_eq = self.prove_equivalence(learned_dfa)
                if not is_eq:
                    assert False

        assert True

    def test_cex_processing(self):
        angluin_example = get_Angluin_dfa()

        alphabet = angluin_example.get_input_alphabet()

        cex_processing = [None, 'longest_prefix', 'rs']
        automata_type = ['dfa', 'mealy', 'moore']

        for automata in automata_type:
            for cex in cex_processing:
                sul = AutomatonSUL(angluin_example)
                eq_oracle = RandomWalkEqOracle(alphabet, sul, 500)

                learned_dfa = run_Lstar(alphabet, sul, eq_oracle, automaton_type=automata,
                                        cache_and_non_det_check=True, cex_processing=cex, print_level=0)

                is_eq = self.prove_equivalence(learned_dfa)
                if not is_eq:
                    assert False

        assert True

    def test_eq_oracles(self):
        angluin_example = get_Angluin_dfa()

        alphabet = angluin_example.get_input_alphabet()

        automata_type = ['dfa', 'mealy', 'moore']

        for automata in automata_type:
            sul = AutomatonSUL(angluin_example)

            random_walk_eq_oracle = RandomWalkEqOracle(alphabet, sul, 5000, reset_after_cex=True)
            state_origin_eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=10, walk_len=50)
            tran_cov_eq_oracle = TransitionFocusOracle(alphabet, sul, num_random_walks=200, walk_len=30,
                                                       same_state_prob=0.3)
            w_method_eq_oracle = WMethodEqOracle(alphabet, sul, max_number_of_states=len(angluin_example.states) + 1)
            wp_method_eq_oracle = WpMethodEqOracle(alphabet, sul, max_number_of_states=len(angluin_example.states) + 1)
            rwp_method_eq_oracle = RandomWpMethodEqOracle(alphabet, sul)
            random_W_method_eq_oracle = RandomWMethodEqOracle(alphabet, sul, walks_per_state=10, walk_len=50)
            bf_exploration_eq_oracle = BreadthFirstExplorationEqOracle(alphabet, sul, 4)
            random_word_eq_oracle = RandomWordEqOracle(alphabet, sul)
            cache_based_eq_oracle = CacheBasedEqOracle(alphabet, sul)
            kWayStateCoverageEqOracle = KWayStateCoverageEqOracle(alphabet, sul)

            oracles = [random_walk_eq_oracle, random_word_eq_oracle,
                       random_W_method_eq_oracle, w_method_eq_oracle,
                       wp_method_eq_oracle, rwp_method_eq_oracle,
                       kWayStateCoverageEqOracle, cache_based_eq_oracle,
                       bf_exploration_eq_oracle, tran_cov_eq_oracle,
                       state_origin_eq_oracle]

            for oracle in oracles:
                sul = AutomatonSUL(angluin_example)
                oracle.sul = sul

                learned_model = run_Lstar(alphabet, sul, oracle, automaton_type=automata,
                                          cache_and_non_det_check=True, cex_processing=None, print_level=0)

                is_eq = self.prove_equivalence(learned_model)
                if not is_eq:
                    print(learned_model)
                    print(oracle, automata)
                    assert False

        assert True

    def test_all_configuration_combinations(self):
        angluin_example = get_Angluin_dfa()

        alphabet = angluin_example.get_input_alphabet()

        automata_type = ['dfa', 'mealy', 'moore']
        closing_strategies = ['shortest_first', 'longest_first', 'single']
        cex_processing = [None, 'longest_prefix', 'rs']
        suffix_closedness = [True, False]
        caching = [True, False]

        for automata in automata_type:
            for closing in closing_strategies:
                for cex in cex_processing:
                    for suffix in suffix_closedness:
                        for cache in caching:
                            sul = AutomatonSUL(angluin_example)

                            random_walk_eq_oracle = RandomWalkEqOracle(alphabet, sul, 5000, reset_after_cex=True)
                            state_origin_eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=10, walk_len=50)
                            tran_cov_eq_oracle = TransitionFocusOracle(alphabet, sul, num_random_walks=200, walk_len=30,
                                                                       same_state_prob=0.3)
                            w_method_eq_oracle = WMethodEqOracle(alphabet, sul,
                                                                 max_number_of_states=len(angluin_example.states))
                            wp_method_eq_oracle = WpMethodEqOracle(alphabet, sul,
                                                                 max_number_of_states=len(angluin_example.states))
                            rwp_method_eq_oracle = RandomWpMethodEqOracle(alphabet, sul)
                            random_W_method_eq_oracle = RandomWMethodEqOracle(alphabet, sul,
                                                                              walks_per_state=10, walk_len=50)
                            bf_exploration_eq_oracle = BreadthFirstExplorationEqOracle(alphabet, sul, 4)
                            random_word_eq_oracle = RandomWordEqOracle(alphabet, sul)
                            cache_based_eq_oracle = CacheBasedEqOracle(alphabet, sul)

                            oracles = [random_walk_eq_oracle, random_word_eq_oracle, random_W_method_eq_oracle,
                                       rwp_method_eq_oracle, cache_based_eq_oracle, bf_exploration_eq_oracle, 
                                       wp_method_eq_oracle, tran_cov_eq_oracle, w_method_eq_oracle,
                                       state_origin_eq_oracle]

                            if not cache:
                                oracles.remove(cache_based_eq_oracle)

                            for oracle in oracles:
                                sul = AutomatonSUL(angluin_example)
                                oracle.sul = sul

                                learned_model = run_Lstar(alphabet, sul, oracle, automaton_type=automata,
                                                          closing_strategy=closing,
                                                          cache_and_non_det_check=cache,
                                                          cex_processing=cex,
                                                          e_set_suffix_closed=suffix,
                                                          print_level=0)

                                is_eq = self.prove_equivalence(learned_model)
                                if not is_eq:
                                    print(oracle, automata)
                                    assert False

        assert True
