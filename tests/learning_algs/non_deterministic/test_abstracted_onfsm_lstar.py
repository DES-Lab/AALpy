import random
import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_abstracted_ONFSM_Lstar
from aalpy.oracles import RandomWalkEqOracle, RandomWordEqOracle
from aalpy.utils import get_benchmark_ONFSM

SEEDS = range(8)


class TestRunAbstractedOnfsmLstar(unittest.TestCase):
    def test_learns_benchmark_onfsm_with_identity_abstraction(self):
        # An empty abstraction mapping means every output maps to itself (see
        # AbstractedNonDetObservationTable.get_abstraction), so this should learn a model
        # equivalent to the plain (non-abstracted) ONFSM learner's result.
        onfsm = get_benchmark_ONFSM()
        alphabet = onfsm.get_input_alphabet()

        for seed in SEEDS:
            random.seed(seed)
            sul = AutomatonSUL(onfsm)
            oracle = RandomWordEqOracle(alphabet, sul, num_walks=200, min_walk_len=2, max_walk_len=5)

            learned_onfsm = run_abstracted_ONFSM_Lstar(alphabet, sul, oracle, abstraction_mapping={},
                                                        n_sampling=20, print_level=0)

            eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=3000, reset_prob=0.09,
                                            reset_after_cex=True)
            cex = eq_oracle.find_cex(learned_onfsm)

            self.assertIsNone(cex, f'seed {seed}: independent oracle found a counterexample')
            self.assertEqual(len(learned_onfsm.states), len(onfsm.states),
                              f'seed {seed}: learned model has wrong number of states')

    def test_real_abstraction_yields_a_model_no_larger_than_the_identity_one(self):
        # Grouping outputs 0 and 2 into the same equivalence class can only merge states, never
        # split them, so the abstracted model should never have more states than the ground truth.
        random.seed(0)
        onfsm = get_benchmark_ONFSM()
        alphabet = onfsm.get_input_alphabet()
        sul = AutomatonSUL(onfsm)
        oracle = RandomWordEqOracle(alphabet, sul, num_walks=200, min_walk_len=2, max_walk_len=5)

        learned_onfsm = run_abstracted_ONFSM_Lstar(alphabet, sul, oracle,
                                                    abstraction_mapping={0: 'low', 2: 'low', 3: 'high'},
                                                    n_sampling=20, print_level=0)

        self.assertLessEqual(len(learned_onfsm.states), len(onfsm.states))

    def test_return_data_reports_consistent_learning_info(self):
        random.seed(0)
        onfsm = get_benchmark_ONFSM()
        alphabet = onfsm.get_input_alphabet()
        sul = AutomatonSUL(onfsm)
        oracle = RandomWordEqOracle(alphabet, sul, num_walks=200, min_walk_len=2, max_walk_len=5)

        learned_onfsm, info = run_abstracted_ONFSM_Lstar(alphabet, sul, oracle, abstraction_mapping={},
                                                          n_sampling=20, print_level=0, return_data=True)

        self.assertEqual(info['automaton_size'], len(learned_onfsm.states))
        self.assertGreaterEqual(info['learning_rounds'], 1)
        self.assertGreaterEqual(info['queries_learning'], 1)


if __name__ == '__main__':
    unittest.main()
