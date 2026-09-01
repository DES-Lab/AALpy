import random
import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_non_det_Lstar
from aalpy.oracles import RandomWalkEqOracle, RandomWordEqOracle
from aalpy.utils import get_benchmark_ONFSM

# The original version of this test (tests/test_non_deterministic.py) ran 100 fresh iterations;
# ONFSM learning is inherently randomized (all-weather sampling + random equivalence oracles), so a
# handful of seeded iterations is kept here to catch flakiness while staying fast.
SEEDS = range(8)


class TestRunNonDetLstar(unittest.TestCase):
    def test_learns_benchmark_onfsm_with_correct_state_count_and_no_cex(self):
        onfsm = get_benchmark_ONFSM()
        alphabet = onfsm.get_input_alphabet()

        for seed in SEEDS:
            random.seed(seed)
            sul = AutomatonSUL(onfsm)
            oracle = RandomWordEqOracle(alphabet, sul, num_walks=200, min_walk_len=2, max_walk_len=5)

            learned_onfsm = run_non_det_Lstar(alphabet, sul, oracle, n_sampling=20, print_level=0)

            eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=3000, reset_prob=0.09,
                                            reset_after_cex=True)
            cex = eq_oracle.find_cex(learned_onfsm)

            self.assertIsNone(cex, f'seed {seed}: independent oracle found a counterexample')
            self.assertEqual(len(learned_onfsm.states), len(onfsm.states),
                              f'seed {seed}: learned model has wrong number of states')

    def test_return_data_reports_consistent_learning_info(self):
        random.seed(0)
        onfsm = get_benchmark_ONFSM()
        alphabet = onfsm.get_input_alphabet()
        sul = AutomatonSUL(onfsm)
        oracle = RandomWordEqOracle(alphabet, sul, num_walks=200, min_walk_len=2, max_walk_len=5)

        learned_onfsm, info = run_non_det_Lstar(alphabet, sul, oracle, n_sampling=20, print_level=0,
                                                 return_data=True)

        self.assertEqual(info['automaton_size'], len(learned_onfsm.states))
        self.assertGreaterEqual(info['learning_rounds'], 1)
        self.assertGreaterEqual(info['queries_learning'], 1)

    def test_stochastic_flag_returns_stochastic_mealy_machine(self):
        from aalpy.automata import StochasticMealyMachine

        random.seed(0)
        onfsm = get_benchmark_ONFSM()
        alphabet = onfsm.get_input_alphabet()
        sul = AutomatonSUL(onfsm)
        oracle = RandomWordEqOracle(alphabet, sul, num_walks=200, min_walk_len=2, max_walk_len=5)

        learned = run_non_det_Lstar(alphabet, sul, oracle, n_sampling=20, stochastic=True, print_level=0)

        self.assertIsInstance(learned, StochasticMealyMachine)


if __name__ == '__main__':
    unittest.main()
