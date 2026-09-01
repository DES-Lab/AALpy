import unittest

from aalpy.automata import MealyMachine, MealyState
from aalpy.base.SUL import CacheSUL
from aalpy.oracles import CacheBasedEqOracle
from aalpy.SULs import AutomatonSUL


def chain_mealy(length, last_output='o'):
    """
    Single-input-letter Mealy chain s0 -a/o-> s1 -a/o-> ... -a/o-> s_length, where the final
    transition into s_length produces last_output instead of 'o'.
    """
    states = [MealyState(f's{i}') for i in range(length + 1)]
    for i in range(length):
        out = last_output if i == length - 1 else 'o'
        states[i].transitions = {'a': states[i + 1]}
        states[i].output_fun = {'a': out}
    states[length].transitions = {'a': states[length]}
    states[length].output_fun = {'a': 'o'}
    mm = MealyMachine(states[0], states)
    mm.compute_prefixes()
    return mm


class CacheBasedEqOracleTests(unittest.TestCase):

    def test_finds_cex_reachable_via_cached_prefix(self):
        reference = chain_mealy(6, last_output='o')
        hypothesis = chain_mealy(6, last_output='x')

        sul = CacheSUL(AutomatonSUL(reference))
        sul.query(('a',) * 5)

        oracle = CacheBasedEqOracle(['a'], sul, num_walks=10, depth_increase=1)
        cex = oracle.find_cex(hypothesis)

        self.assertIsNotNone(cex)
        reference.reset_to_initial()
        hypothesis.reset_to_initial()
        sul_out = [reference.step(i) for i in cex]
        hyp_out = [hypothesis.step(i) for i in cex]
        self.assertNotEqual(sul_out[-1], hyp_out[-1])
        self.assertEqual(sul_out[:-1], hyp_out[:-1])

    def test_does_not_find_cex_when_difference_is_uncached_and_out_of_reach(self):
        reference = chain_mealy(6, last_output='o')
        hypothesis = chain_mealy(6, last_output='x')

        sul = CacheSUL(AutomatonSUL(reference))
        # nothing has been queried yet, so the only cached "leaf" is the empty prefix
        oracle = CacheBasedEqOracle(['a'], sul, num_walks=10, depth_increase=1)
        cex = oracle.find_cex(hypothesis)

        self.assertIsNone(cex)

    def test_no_cex_for_equivalent_hypothesis(self):
        reference = chain_mealy(4, last_output='o')
        hypothesis = chain_mealy(4, last_output='o')

        sul = CacheSUL(AutomatonSUL(reference))
        sul.query(('a', 'a', 'a'))

        oracle = CacheBasedEqOracle(['a'], sul, num_walks=20, depth_increase=3)
        self.assertIsNone(oracle.find_cex(hypothesis))

    def test_get_paths_collects_all_leaves(self):
        reference = chain_mealy(3)
        sul = CacheSUL(AutomatonSUL(reference))
        sul.query(('a', 'a'))

        oracle = CacheBasedEqOracle(['a'], sul)
        paths = oracle.get_paths(sul.cache.root_node)
        self.assertEqual(paths, [['a', 'a']])


if __name__ == '__main__':
    unittest.main()
