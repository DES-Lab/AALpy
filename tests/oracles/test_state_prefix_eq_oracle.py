import random
import unittest

from aalpy.automata import MealyMachine, MealyState
from aalpy.oracles import StatePrefixEqOracle
from aalpy.SULs import AutomatonSUL


def branching_mealy():
    """
    4-state Mealy machine over {a, b}. From s0, 'a' reaches s1 and 'b' reaches s2; s1 and s2 both lead to s3 on 'a'.
    s3's outgoing transitions are where the two variants below differ.
    """
    s0, s1, s2, s3 = (MealyState(f's{i}') for i in range(4))
    s0.transitions = {'a': s1, 'b': s2}
    s0.output_fun = {'a': 'o', 'b': 'o'}
    s1.transitions = {'a': s3, 'b': s1}
    s1.output_fun = {'a': 'o', 'b': 'o'}
    s2.transitions = {'a': s3, 'b': s2}
    s2.output_fun = {'a': 'o', 'b': 'o'}
    s3.transitions = {'a': s3, 'b': s3}
    s3.output_fun = {'a': 'o', 'b': 'o'}
    mm = MealyMachine(s0, [s0, s1, s2, s3])
    mm.compute_prefixes()
    return mm, (s0, s1, s2, s3)


class StatePrefixEqOracleTests(unittest.TestCase):

    def test_finds_difference_reachable_only_via_specific_state_suffix(self):
        # difference is only observable from s2's own 'b' self-loop, unreachable from s0/s1's local suffixes alone
        reference, (s0, s1, s2, s3) = branching_mealy()
        hypothesis, (h0, h1, h2, h3) = branching_mealy()
        h2.output_fun['b'] = 'x'

        random.seed(0)
        oracle = StatePrefixEqOracle(['a', 'b'], AutomatonSUL(reference), walks_per_state=30, walk_len=3)
        cex = oracle.find_cex(hypothesis)

        self.assertIsNotNone(cex)
        reference.reset_to_initial()
        hypothesis.reset_to_initial()
        sul_out = [reference.step(i) for i in cex]
        hyp_out = [hypothesis.step(i) for i in cex]
        self.assertNotEqual(sul_out[-1], hyp_out[-1])
        self.assertEqual(sul_out[:-1], hyp_out[:-1])
        # the divergent suffix step must be taken from s2 (reached only via prefix 'b')
        self.assertEqual(cex[0], 'b')

    def test_no_cex_for_equivalent_hypothesis(self):
        random.seed(0)
        reference, _ = branching_mealy()
        hypothesis, _ = branching_mealy()

        oracle = StatePrefixEqOracle(['a', 'b'], AutomatonSUL(reference), walks_per_state=30, walk_len=5)
        self.assertIsNone(oracle.find_cex(hypothesis))

    def test_max_tests_bounds_number_of_queries(self):
        random.seed(0)
        reference, _ = branching_mealy()
        hypothesis, _ = branching_mealy()

        oracle = StatePrefixEqOracle(['a', 'b'], AutomatonSUL(reference), walks_per_state=30, walk_len=5,
                                     max_tests=5)
        oracle.find_cex(hypothesis)
        self.assertLessEqual(oracle.num_queries, 5)

    def test_zero_walks_per_state_never_tests_anything(self):
        reference, _ = branching_mealy()
        hypothesis, (h0, h1, h2, h3) = branching_mealy()
        h2.output_fun['b'] = 'x'

        oracle = StatePrefixEqOracle(['a', 'b'], AutomatonSUL(reference), walks_per_state=0, walk_len=5)
        self.assertIsNone(oracle.find_cex(hypothesis))
        self.assertEqual(oracle.num_queries, 0)


if __name__ == '__main__':
    unittest.main()
