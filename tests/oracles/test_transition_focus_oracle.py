import random
import unittest

from aalpy.automata import MealyMachine, MealyState
from aalpy.oracles import TransitionFocusOracle
from aalpy.SULs import AutomatonSUL


def self_loop_vs_transition_mealy():
    """
    2-state Mealy machine. s0 --a/o1--> s0 (self loop) and s0 --b/o2--> s1 (different-state transition).
    s1 mirrors s0's transitions so the walk never gets stuck.
    """
    s0, s1 = MealyState('s0'), MealyState('s1')
    s0.transitions = {'a': s0, 'b': s1}
    s0.output_fun = {'a': 'o1', 'b': 'o2'}
    s1.transitions = {'a': s1, 'b': s0}
    s1.output_fun = {'a': 'o1', 'b': 'o2'}
    mm = MealyMachine(s0, [s0, s1])
    mm.compute_prefixes()
    return mm, s0, s1


class TransitionFocusOracleTests(unittest.TestCase):

    def test_same_state_prob_1_finds_self_loop_only_difference(self):
        reference, s0, s1 = self_loop_vs_transition_mealy()
        hypothesis, h0, h1 = self_loop_vs_transition_mealy()
        h0.output_fun['a'] = 'x'  # only the self-loop on s0 differs

        oracle = TransitionFocusOracle(['a', 'b'], AutomatonSUL(reference), num_random_walks=1, walk_len=1,
                                       same_state_prob=1.0)
        cex = oracle.find_cex(hypothesis)

        self.assertEqual(tuple(cex), ('a',))

    def test_same_state_prob_0_never_probes_the_self_loop(self):
        reference, s0, s1 = self_loop_vs_transition_mealy()
        hypothesis, h0, h1 = self_loop_vs_transition_mealy()
        h0.output_fun['a'] = 'x'  # only the self-loop on s0 differs, 'b' transitions remain correct everywhere

        oracle = TransitionFocusOracle(['a', 'b'], AutomatonSUL(reference), num_random_walks=20, walk_len=10,
                                       same_state_prob=0.0)
        self.assertIsNone(oracle.find_cex(hypothesis))

    def test_finds_difference_on_diff_state_transition_over_several_seeds(self):
        successes = 0
        for seed in range(10):
            random.seed(seed)
            reference, s0, s1 = self_loop_vs_transition_mealy()
            hypothesis, h0, h1 = self_loop_vs_transition_mealy()
            h0.output_fun['b'] = 'x'  # only the s0 -> s1 transition differs

            oracle = TransitionFocusOracle(['a', 'b'], AutomatonSUL(reference), num_random_walks=30, walk_len=10,
                                           same_state_prob=0.2)
            cex = oracle.find_cex(hypothesis)
            if cex is not None:
                successes += 1

        self.assertGreaterEqual(successes, 9)

    def test_no_cex_for_equivalent_hypothesis(self):
        random.seed(0)
        reference, _, _ = self_loop_vs_transition_mealy()
        hypothesis, _, _ = self_loop_vs_transition_mealy()

        oracle = TransitionFocusOracle(['a', 'b'], AutomatonSUL(reference), num_random_walks=50, walk_len=10)
        self.assertIsNone(oracle.find_cex(hypothesis))


if __name__ == '__main__':
    unittest.main()
