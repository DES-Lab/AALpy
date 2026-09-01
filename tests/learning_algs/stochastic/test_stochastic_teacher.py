import random
import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Mdp, MdpState
from aalpy.learning_algs.stochastic.DifferenceChecker import HoeffdingChecker
from aalpy.learning_algs.stochastic.StochasticTeacher import Node, StochasticTeacher


def _branching_mdp():
    """
    s0(A) --a--> s1(B) with prob 0.7, s2(C) with prob 0.3; s0 --b--> s0(A) with prob 1.0.
    s1, s2 --a/b--> s0 with prob 1.0 (absorbing back).
    """
    s0 = MdpState('s0', output='A')
    s1 = MdpState('s1', output='B')
    s2 = MdpState('s2', output='C')
    s0.transitions['a'].append((s1, 0.7))
    s0.transitions['a'].append((s2, 0.3))
    s0.transitions['b'].append((s0, 1.0))
    s1.transitions['a'].append((s0, 1.0))
    s1.transitions['b'].append((s0, 1.0))
    s2.transitions['a'].append((s0, 1.0))
    s2.transitions['b'].append((s0, 1.0))
    return Mdp(s0, [s0, s1, s2])


class _NullEqOracle:
    num_queries = 0
    num_steps = 0

    def reset_counter(self):
        pass

    def find_cex(self, hypothesis):
        return None


class StochasticTeacherAddAndQueryTest(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        self.sul = AutomatonSUL(_branching_mdp())
        self.teacher = StochasticTeacher(self.sul, n_c=5, eq_oracle=_NullEqOracle(),
                                          automaton_type='mdp', compatibility_checker=HoeffdingChecker())

    def test_initial_value_is_captured_from_sul(self):
        self.assertEqual(self.teacher.initial_value, ['A'])
        self.assertEqual(self.teacher.root_node.output, 'A')

    def test_add_updates_frequency_and_children(self):
        self.teacher.back_to_root()
        self.teacher.add('a', 'B')
        self.assertEqual(self.teacher.root_node.input_frequencies['a'], 1)
        child = self.teacher.root_node.get_child('a', 'B')
        self.assertIsNotNone(child)
        self.assertEqual(child.frequency, 1)

        self.teacher.back_to_root()
        self.teacher.add('a', 'B')
        # same (input, output) pair increments the same child's frequency
        self.assertEqual(self.teacher.root_node.input_frequencies['a'], 2)
        self.assertEqual(child.frequency, 2)

    def test_frequency_query_reflects_added_traces(self):
        for _ in range(5):
            self.teacher.back_to_root()
            self.teacher.add('a', 'B')
        for _ in range(3):
            self.teacher.back_to_root()
            self.teacher.add('a', 'C')

        # s is the mdp-format prefix (initial_output,), e is ('a',)
        freq = self.teacher.frequency_query(('A',), ('a',))
        self.assertEqual(freq, {'B': 5, 'C': 3})

    def test_frequency_query_missing_path_returns_empty_dict(self):
        freq = self.teacher.frequency_query(('A', 'a', 'B'), ('a',))
        self.assertEqual(freq, {})

    def test_complete_query_false_below_n_c_true_at_or_above(self):
        for i in range(4):
            self.teacher.back_to_root()
            self.teacher.add('a', 'B')
        self.assertFalse(self.teacher.complete_query(('A',), ('a',)))

        self.teacher.back_to_root()
        self.teacher.add('a', 'B')
        self.assertTrue(self.teacher.complete_query(('A',), ('a',)))

    def test_complete_query_is_cached(self):
        for i in range(5):
            self.teacher.back_to_root()
            self.teacher.add('a', 'B')
        self.assertTrue(self.teacher.complete_query(('A',), ('a',)))
        # the mdp's leading initial-output symbol is stripped internally before building the cache key
        self.assertIn(('a',), self.teacher.complete_query_cache)


class StochasticTeacherTreeQueryTest(unittest.TestCase):

    def setUp(self):
        random.seed(1)
        self.sul = AutomatonSUL(_branching_mdp())
        self.teacher = StochasticTeacher(self.sul, n_c=5, eq_oracle=_NullEqOracle(),
                                          automaton_type='mdp', compatibility_checker=HoeffdingChecker())

    def test_tree_query_samples_from_pta_and_adds_to_tree(self):
        pta_root = Node('A')
        pta_root.input_frequencies['a'] = 10
        pta_root.children['a']['B'] = Node('B')
        pta_root.children['a']['C'] = Node('C')

        for _ in range(50):
            self.teacher.tree_query(pta_root)

        total_a = self.teacher.root_node.get_frequency_sum('a')
        self.assertEqual(total_a, 50)
        freqs = self.teacher.root_node.get_output_frequencies('a')
        self.assertEqual(sum(freqs.values()), 50)
        self.assertTrue(set(freqs.keys()).issubset({'B', 'C'}))


class StochasticTeacherEquivalenceQueryTest(unittest.TestCase):

    def setUp(self):
        random.seed(7)
        self.sul = AutomatonSUL(_branching_mdp())

    def test_last_cex_is_reused_while_still_valid(self):
        teacher = StochasticTeacher(self.sul, n_c=5, eq_oracle=_NullEqOracle(),
                                     automaton_type='mdp', compatibility_checker=HoeffdingChecker())
        hyp_s0 = MdpState('s0', output='A')
        hyp_s0.prefix = ('A',)
        hypothesis = Mdp(hyp_s0, [hyp_s0])
        # hypothesis has no transitions at all, so any cex trying to step 'a' is still "unprocessed";
        # cex format is (o0, i1, o1, ..., i_n), ending in a dangling input
        teacher.last_cex = ('A', 'a', 'B', 'a')

        cex = teacher.equivalence_query(hypothesis)
        self.assertEqual(cex, ('A', 'a', 'B', 'a'))

    def test_falls_back_to_eq_oracle_when_no_cached_or_tree_cex(self):
        class _OracleWithCex:
            num_queries = 0
            num_steps = 0

            def reset_counter(self):
                pass

            def find_cex(self, hypothesis):
                return ('A', 'a', 'B', 'a', 'A')

        teacher = StochasticTeacher(self.sul, n_c=5, eq_oracle=_OracleWithCex(),
                                     automaton_type='mdp', compatibility_checker=HoeffdingChecker())
        hyp_s0 = MdpState('s0', output='A')
        hyp_s0.prefix = ('A',)
        hypothesis = Mdp(hyp_s0, [hyp_s0])

        cex = teacher.equivalence_query(hypothesis)
        # equivalence_query strips the last element of the oracle's cex
        self.assertEqual(cex, ('A', 'a', 'B', 'a'))
        self.assertEqual(teacher.last_cex, cex)

    def test_bfs_for_cex_in_tree_finds_output_mismatch(self):
        teacher = StochasticTeacher(self.sul, n_c=5, eq_oracle=_NullEqOracle(),
                                     automaton_type='mdp', compatibility_checker=HoeffdingChecker())
        for _ in range(10):
            teacher.back_to_root()
            teacher.add('a', 'B')

        hyp_s0 = MdpState('s0', output='A')
        hyp_s0.prefix = ('A',)
        hypothesis = Mdp(hyp_s0, [hyp_s0])  # no transitions at all for 'a'

        cex = teacher.bfs_for_cex_in_tree(hypothesis)
        self.assertEqual(cex, ('A', 'a'))


if __name__ == '__main__':
    unittest.main()
