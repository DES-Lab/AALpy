import unittest

from aalpy.automata import Onfsm, OnfsmState
from aalpy.learning_algs.non_deterministic.TraceTree import TraceTree


def branching_onfsm():
    s0 = OnfsmState('s0')
    s1 = OnfsmState('s1')
    s2 = OnfsmState('s2')
    s0.transitions['a'].append(('x', s1))
    s0.transitions['a'].append(('y', s2))
    s1.transitions['a'].append(('x', s0))
    s2.transitions['a'].append(('y', s0))
    return Onfsm(s0, [s0, s1, s2])


class TestTraceTreeConstruction(unittest.TestCase):
    def test_new_tree_has_root_and_no_cursor(self):
        tree = TraceTree()
        self.assertIsNone(tree.root_node.output)
        self.assertIsNone(tree.curr_node)

    def test_reset_sets_cursor_to_root(self):
        tree = TraceTree()
        tree.reset()
        self.assertIs(tree.curr_node, tree.root_node)


class TestAddToTree(unittest.TestCase):
    def test_add_to_tree_creates_child_and_moves_cursor(self):
        tree = TraceTree()
        tree.reset()
        tree.add_to_tree('a', 'x')
        self.assertIsNot(tree.curr_node, tree.root_node)
        self.assertEqual(tree.curr_node.output, 'x')
        self.assertIs(tree.curr_node.parent, tree.root_node)

    def test_repeated_pair_reuses_node_and_bumps_frequency(self):
        tree = TraceTree()
        tree.reset()
        tree.add_to_tree('a', 'x')
        node_first = tree.curr_node
        tree.reset()
        tree.add_to_tree('a', 'x')
        self.assertIs(tree.curr_node, node_first)
        self.assertEqual(node_first.frequency_counter, 2)

    def test_different_output_for_same_input_creates_sibling(self):
        tree = TraceTree()
        tree.reset()
        tree.add_to_tree('a', 'x')
        tree.reset()
        tree.add_to_tree('a', 'y')
        self.assertEqual(len(tree.root_node.children['a']), 2)
        outputs = {child.output for child in tree.root_node.children['a']}
        self.assertEqual(outputs, {'x', 'y'})


class TestAddTrace(unittest.TestCase):
    def test_add_trace_resets_before_inserting(self):
        tree = TraceTree()
        tree.reset()
        tree.add_to_tree('a', 'x')
        tree.add_trace(('a',), ('y',))
        # add_trace resets the cursor to root first, so 'y' is a sibling of 'x', not a child of it
        self.assertEqual(len(tree.root_node.children['a']), 2)

    def test_add_trace_builds_full_path(self):
        tree = TraceTree()
        tree.add_trace(('a', 'a'), ('x', 'y'))
        node = tree.get_to_node(('a', 'a'), ('x', 'y'))
        self.assertIsNotNone(node)
        self.assertEqual(node.get_prefix(), ('x', 'y'))


class TestGetToNode(unittest.TestCase):
    def test_returns_node_on_known_path(self):
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        node = tree.get_to_node(('a',), ('x',))
        self.assertIsNotNone(node)

    def test_returns_none_on_unknown_output(self):
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        self.assertIsNone(tree.get_to_node(('a',), ('z',)))

    def test_returns_root_for_empty_path(self):
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        self.assertIs(tree.get_to_node((), ()), tree.root_node)


class TestGetAllTraces(unittest.TestCase):
    def test_returns_empty_list_when_prefix_is_empty_tuple(self):
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        self.assertEqual(tree.get_all_traces((), ('a',)), [])

    def test_returns_empty_list_when_suffix_is_empty(self):
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        self.assertEqual(tree.get_all_traces(((), ()), ()), [])

    def test_root_prefix_is_a_2tuple_of_empty_tuples_and_is_queried_normally(self):
        # ((), ()) is truthy as a whole (it's a non-empty 2-tuple), so the "not prefix" guard
        # in get_all_traces never triggers for it - this is the standard shape of an S-set row.
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        self.assertEqual(tree.get_all_traces(((), ()), ('a',)), [('x',)])

    def test_branches_are_all_returned_for_the_same_prefix(self):
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        tree.add_trace(('a',), ('y',))
        traces = tree.get_all_traces(((), ()), ('a',))
        self.assertCountEqual(traces, [('x',), ('y',)])

    def test_returns_only_the_suffix_portion_of_the_trace(self):
        tree = TraceTree()
        tree.add_trace(('a', 'b'), ('x', 'z'))
        # prefix identifies the node reached after 'a'/'x', then we trace 'b' from there
        traces = tree.get_all_traces((('a',), ('x',)), ('b',))
        self.assertEqual(traces, [('z',)])

    def test_returns_empty_list_for_unknown_prefix(self):
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        self.assertEqual(tree.get_all_traces((('a',), ('z',)), ('a',)), [])

    def test_returns_empty_list_when_suffix_input_never_observed(self):
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        self.assertEqual(tree.get_all_traces(((), ()), ('b',)), [])


class TestGetTable(unittest.TestCase):
    def test_get_table_matches_get_all_traces_per_cell(self):
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        tree.add_trace(('a',), ('y',))
        s = [((), ())]
        e = [('a',)]
        table = tree.get_table(s, e)
        self.assertCountEqual(table[s[0]][e[0]], tree.get_all_traces(s[0], e[0]))


class TestSamplingFrequencyAndDistribution(unittest.TestCase):
    def test_frequency_counts_repeated_samples(self):
        tree = TraceTree()
        for _ in range(3):
            tree.reset()
            tree.add_to_tree('a', 'x')
        for _ in range(2):
            tree.reset()
            tree.add_to_tree('a', 'y')
        self.assertEqual(tree.get_s_e_sampling_frequency(((), ()), ('a',)), 5)

    def test_frequency_is_zero_for_never_sampled_path(self):
        tree = TraceTree()
        tree.add_trace(('a',), ('x',))
        self.assertEqual(tree.get_s_e_sampling_frequency(((), ()), ('b',)), 0)

    def test_frequency_over_multi_step_suffix(self):
        tree = TraceTree()
        for _ in range(4):
            tree.reset()
            tree.add_to_tree('a', 'x')
            tree.add_to_tree('b', 'z')
        self.assertEqual(tree.get_s_e_sampling_frequency(((), ()), ('a', 'b')), 4)

    def test_sampling_distribution_matches_observed_ratios(self):
        tree = TraceTree()
        for _ in range(3):
            tree.reset()
            tree.add_to_tree('a', 'x')
        for _ in range(1):
            tree.reset()
            tree.add_to_tree('a', 'y')
        distribution = tree.get_sampling_distributions(((), ()), 'a')
        self.assertAlmostEqual(distribution['x'], 0.75)
        self.assertAlmostEqual(distribution['y'], 0.25)
        self.assertAlmostEqual(sum(distribution.values()), 1.0)


class TestFindCexInCache(unittest.TestCase):
    def test_returns_none_when_cache_agrees_with_hypothesis(self):
        onfsm = branching_onfsm()
        tree = TraceTree()
        tree.add_trace(('a', 'a'), ('x', 'x'))
        tree.add_trace(('a', 'a'), ('y', 'y'))
        self.assertIsNone(tree.find_cex_in_cache(onfsm))

    def test_finds_cex_when_cache_disagrees_with_hypothesis(self):
        onfsm = branching_onfsm()
        tree = TraceTree()
        # from s1 (reached via 'a'/'x'), only 'x' is a valid onward output, so 'y' is a cex
        tree.add_trace(('a', 'a'), ('x', 'y'))
        cex = tree.find_cex_in_cache(onfsm)
        self.assertEqual(cex, (['a', 'a'], ['x', 'y']))


if __name__ == '__main__':
    unittest.main()
